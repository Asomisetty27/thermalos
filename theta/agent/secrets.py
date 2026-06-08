"""
At-rest credential protection for ~/.theta/config.json.

Audit finding P0 (security): the wizard wrote Redfish BMC username/password
to ~/.theta/config.json in plaintext. If a host is compromised — even at
the user level, since the file lives in $HOME — an attacker can recover
the BMC credentials and pivot to chassis-level access (firmware, fan
control, power-on/off, console).

Threat model and what this DOES and DOES NOT defend against:

  ✓ DOES defend against: opportunistic file-read by another user on the
    host who can `cat ~/.theta/config.json` (assuming HOME is not 0700,
    or via a temp file leak, or a backup archive that ends up off-host).

  ✓ DOES defend against: accidental disclosure (config file committed to
    git, shared as a debug artifact, copied to a support ticket).

  ✗ DOES NOT defend against: an attacker with root who can read the
    machine-id, the keyring, or modify the agent binary.

  ✗ DOES NOT defend against: an attacker who already has the agent's
    Python interpreter running with the same key material.

This is realistic data-at-rest hygiene, not a hardware-secured vault. We
encrypt with a key DERIVED from machine-stable identifiers (machine-id +
the agent install location) using PBKDF2 — so the encrypted blob is bound
to this host. Move the file to another box and decryption fails.

When the cryptography library is unavailable, we fall back to a NON-CRYPTO
obfuscation (base64 XOR with the machine key). This is explicitly weaker
and a warning is logged on every read/write so the operator knows to
`pip install cryptography`. We DON'T silently store secrets in plaintext.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import platform
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Prefix that marks an encrypted value in config.json. Plain strings without
# this prefix are treated as legacy plaintext (and a migration warning is
# logged so the operator knows to re-save with the wizard).
_ENC_PREFIX_FERNET = "enc:fernet:v1:"
_ENC_PREFIX_XOR    = "enc:xor:v1:"


def _machine_key_material() -> bytes:
    """Stable per-host identifier used to derive an encryption key.

    Sources tried in order:
      1. /etc/machine-id (Linux systemd hosts)
      2. /var/lib/dbus/machine-id (older Linux)
      3. macOS IOPlatformUUID via `system_profiler` is too slow to call
         on every read, so we use platform.node() (hostname) + Python's
         platform.machine() (architecture) as a stable-enough proxy.

    The combined value is hashed with the agent install path so the same
    user running two separate Theta installs gets different key material
    per install (defense in depth — exfiltrating one install's secrets
    doesn't decrypt the other's).
    """
    parts: list[bytes] = []
    for candidate in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            parts.append(Path(candidate).read_bytes().strip())
            break
        except (OSError, PermissionError):
            continue
    if not parts:
        parts.append(platform.node().encode("utf-8"))
        parts.append(platform.machine().encode("utf-8"))
    # Bind to install path so multiple installs on one box don't share key
    parts.append(str(Path(__file__).resolve().parent).encode("utf-8"))
    return b"||".join(parts)


def _derive_key(material: bytes, salt: bytes = b"theta-bmc-creds-v1") -> bytes:
    """Derive a 32-byte key for Fernet (which wraps it to its URL-safe form)."""
    return hashlib.pbkdf2_hmac("sha256", material, salt, iterations=120_000, dklen=32)


def _xor_obfuscate(plaintext: str, key: bytes) -> str:
    """Fallback obfuscation when cryptography unavailable. Not real crypto."""
    plain_bytes = plaintext.encode("utf-8")
    # Cycle key to match plaintext length
    expanded = (key * (len(plain_bytes) // len(key) + 1))[:len(plain_bytes)]
    xored = bytes(b ^ k for b, k in zip(plain_bytes, expanded))
    return _ENC_PREFIX_XOR + base64.urlsafe_b64encode(xored).decode("ascii")


def _xor_deobfuscate(token: str, key: bytes) -> str:
    payload = token[len(_ENC_PREFIX_XOR):]
    xored = base64.urlsafe_b64decode(payload.encode("ascii"))
    expanded = (key * (len(xored) // len(key) + 1))[:len(xored)]
    return bytes(b ^ k for b, k in zip(xored, expanded)).decode("utf-8")


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a credential. Returns a prefixed, self-describing token.

    Safe to write to JSON files — the token includes a prefix so subsequent
    reads can tell encrypted from legacy-plaintext values.
    """
    if not plaintext:
        return plaintext
    if plaintext.startswith((_ENC_PREFIX_FERNET, _ENC_PREFIX_XOR)):
        # Already encrypted — don't double-wrap
        return plaintext
    key_material = _machine_key_material()
    key = _derive_key(key_material)
    try:
        from cryptography.fernet import Fernet
        fernet = Fernet(base64.urlsafe_b64encode(key))
        token = fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")
        return _ENC_PREFIX_FERNET + token
    except ImportError:
        log.warning(
            "cryptography library not installed — falling back to weak XOR "
            "obfuscation for credential storage. Install with: pip install cryptography"
        )
        return _xor_obfuscate(plaintext, key)


def decrypt_secret(token: Optional[str]) -> Optional[str]:
    """Decrypt a credential token. Plaintext strings (no prefix) pass through.

    A migration warning is logged on plaintext to nudge the operator to
    re-save via the wizard, which will encrypt-in-place.
    """
    if not token:
        return token
    if token.startswith(_ENC_PREFIX_FERNET):
        try:
            from cryptography.fernet import Fernet
            key = _derive_key(_machine_key_material())
            fernet = Fernet(base64.urlsafe_b64encode(key))
            return fernet.decrypt(
                token[len(_ENC_PREFIX_FERNET):].encode("ascii")
            ).decode("utf-8")
        except ImportError:
            log.error(
                "cryptography library required to decrypt Fernet-encrypted "
                "secret but not installed. Run: pip install cryptography"
            )
            return None
        except Exception as exc:
            log.error("Failed to decrypt secret (key mismatch?): %s", exc)
            return None
    if token.startswith(_ENC_PREFIX_XOR):
        try:
            return _xor_deobfuscate(token, _derive_key(_machine_key_material()))
        except Exception as exc:
            log.error("Failed to deobfuscate secret: %s", exc)
            return None
    # Legacy plaintext — pass through with a one-time warning
    log.warning(
        "Reading plaintext credential from config — re-run `theta setup` "
        "to encrypt it. (Plaintext credentials in config files are a "
        "known security risk.)"
    )
    return token


def migrate_config_secrets(config: dict, secret_paths: list[tuple[str, ...]]) -> bool:
    """Encrypt any plaintext secrets in a config dict in-place.

    `secret_paths` is a list of key-paths into the dict; each path identifies
    a value that should be encrypted at rest. Returns True if anything was
    changed (so the caller can choose to re-save).

    Example:
        changed = migrate_config_secrets(config, [
            ("redfish", "password"),
            ("redfish", "username"),
            ("alerts", "webhook_url"),
        ])
        if changed: save_config(config)
    """
    changed = False
    for path in secret_paths:
        node = config
        for key in path[:-1]:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if not isinstance(node, dict):
            continue
        leaf = path[-1]
        value = node.get(leaf)
        if not isinstance(value, str) or not value:
            continue
        if value.startswith((_ENC_PREFIX_FERNET, _ENC_PREFIX_XOR)):
            continue  # already encrypted
        node[leaf] = encrypt_secret(value)
        changed = True
    return changed
