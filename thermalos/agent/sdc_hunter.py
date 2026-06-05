"""
Silent Data Corruption Hunter — Meta Ripple pattern as OSS.

Runs GPU validation kernels during idle windows. Detects bit-flip corruption
that produces wrong answers without raising any hardware error. SDC is the
failure mode that silently corrupts weeks of LLM training — model diverges
slowly, results are invalid, nobody knows until the loss curve looks wrong.

Protocol (Meta Ripple-inspired):
  1. Generate a deterministic input tensor (seeded, reproducible)
  2. Run matmul + reduce on the target GPU
  3. Compare the output hash against the reference hash for that seed
  4. Reference hash is established on first run (assumed clean) and verified
     against all other GPUs in the fleet for cross-validation
  5. Mismatch → SDC event, CRITICAL alert

Detection runs only during idle windows (util < 5%, power < 25W) to avoid
interfering with production workloads.

Cadence:
  - First run: immediately on first idle window after startup
  - Subsequent runs: every HUNT_INTERVAL_S (default: 3600 = 1 hour)
  - Accelerated: every 300s if ECC single-bit rate is rising

Dependencies: torch must be installed on the host (it is, on any ML system).
The validation kernel runs in a subprocess so torch startup cost stays out
of the agent's import path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .metrics import GPUState, AlertEvent

log = logging.getLogger(__name__)

HUNT_INTERVAL_S     = 3600   # normal cadence
HUNT_INTERVAL_FAST  = 300    # accelerated when ECC sbit rising
IDLE_UTIL_THRESHOLD = 5.0    # GPU must be this idle
IDLE_POWER_W        = 25.0   # and this low-power

# Validation kernel: deterministic matmul + reduce with known seed
_VALIDATION_SCRIPT = '''
import sys, json, hashlib
import torch

gpu_idx = int(sys.argv[1])
seed    = int(sys.argv[2])
device  = f"cuda:{gpu_idx}"

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
A = torch.randn(1024, 1024, device=device, dtype=torch.float32)
B = torch.randn(1024, 1024, device=device, dtype=torch.float32)
C = torch.matmul(A, B)
s = C.sum().item()
# Hash the full matrix to catch bit flips anywhere, not just the sum
buf = C.cpu().numpy().tobytes()
h = hashlib.sha256(buf).hexdigest()
print(json.dumps({"gpu": gpu_idx, "seed": seed, "sum": round(s, 6), "hash": h}))
'''


@dataclass
class SDCResult:
    gpu_index:  int
    seed:       int
    passed:     bool
    expected_hash: Optional[str]
    actual_hash:   Optional[str]
    error:      Optional[str] = None


class SDCHunter:
    """
    Per-fleet SDC validation runner.

    Writes the validation script to a temp file once, then spawns it per GPU.
    Cross-GPU validation: if GPU 0 and GPU 1 produce different hashes for the
    same seed, at least one is corrupted.
    """

    def __init__(self, gpu_indices: Optional[list[int]] = None):
        self._gpu_indices      = gpu_indices
        self._reference_hashes: dict[int, dict[int, str]] = {}   # seed → gpu → hash
        self._last_hunt:        dict[int, float] = {}
        self._script_path:      Optional[Path]   = None
        self._sdc_count:        dict[int, int]   = {}

    def _ensure_script(self) -> Path:
        if self._script_path and self._script_path.exists():
            return self._script_path
        tmp = Path(tempfile.mktemp(suffix="_thermalos_sdc.py"))
        tmp.write_text(_VALIDATION_SCRIPT)
        self._script_path = tmp
        return tmp

    async def _run_validation(self, gpu_index: int, seed: int) -> SDCResult:
        """Run the validation kernel on one GPU. Returns the result."""
        script = self._ensure_script()
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(script), str(gpu_index), str(seed),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)

            if proc.returncode != 0:
                err = stderr.decode()[:200]
                if "CUDA" not in err and "torch" not in err.lower():
                    err = "GPU may be inaccessible"
                return SDCResult(gpu_index=gpu_index, seed=seed, passed=False,
                                 expected_hash=None, actual_hash=None, error=err)

            result = json.loads(stdout.decode().strip())
            actual_hash = result["hash"]
            return SDCResult(
                gpu_index    = gpu_index,
                seed         = seed,
                passed       = True,
                expected_hash= None,
                actual_hash  = actual_hash,
            )
        except asyncio.TimeoutError:
            return SDCResult(gpu_index=gpu_index, seed=seed, passed=False,
                             expected_hash=None, actual_hash=None,
                             error="validation timed out after 30s")
        except Exception as e:
            return SDCResult(gpu_index=gpu_index, seed=seed, passed=False,
                             expected_hash=None, actual_hash=None, error=str(e))

    def _is_due(self, gpu_index: int, ecc_sbit_rising: bool) -> bool:
        interval = HUNT_INTERVAL_FAST if ecc_sbit_rising else HUNT_INTERVAL_S
        return time.time() - self._last_hunt.get(gpu_index, 0.0) >= interval

    async def hunt(
        self,
        gpu_states: dict[int, GPUState],
        gpu_util:   dict[int, float],
        gpu_power:  dict[int, float],
        timestamp:  float,
        ecc_sbit_rising: bool = False,
    ) -> list[AlertEvent]:
        """
        Run validation on all idle GPUs that are due for a check.
        Returns list of AlertEvents (one per SDC detected, usually empty).
        """
        candidates = [
            g for g, state in gpu_states.items()
            if state in (GPUState.CLEAN_IDLE,)
            and gpu_util.get(g, 100.0) < IDLE_UTIL_THRESHOLD
            and gpu_power.get(g, 100.0) < IDLE_POWER_W
            and self._is_due(g, ecc_sbit_rising)
        ]

        if not candidates:
            return []

        # Choose a seed that changes daily — consistent within a day for cross-GPU comparison
        seed = int(timestamp // 86400)

        results = await asyncio.gather(
            *(self._run_validation(g, seed) for g in candidates),
            return_exceptions=True,
        )

        alerts = []
        valid_results: list[SDCResult] = [
            r for r in results
            if isinstance(r, SDCResult) and r.passed and r.actual_hash
        ]

        for r in results:
            if isinstance(r, SDCResult):
                self._last_hunt[r.gpu_index] = timestamp

        # Cross-GPU validation: all GPUs with same seed should produce same hash
        if len(valid_results) >= 2:
            hashes = {r.gpu_index: r.actual_hash for r in valid_results}
            all_same = len(set(hashes.values())) == 1

            if not all_same:
                # Find which GPUs deviate from the majority hash
                from collections import Counter
                majority_hash = Counter(hashes.values()).most_common(1)[0][0]

                for gpu_idx, h in hashes.items():
                    if h != majority_hash:
                        self._sdc_count[gpu_idx] = self._sdc_count.get(gpu_idx, 0) + 1
                        log.error("SDC detected on GPU %d (hash mismatch)", gpu_idx)
                        alerts.append(AlertEvent(
                            gpu_index       = gpu_idx,
                            timestamp       = timestamp,
                            state           = GPUState.CRITICAL,
                            prev_state      = GPUState.CLEAN_IDLE,
                            rtheta          = None,
                            rtheta_baseline = None,
                            drift_sigma     = None,
                            confidence      = 0.95,
                            message         = (
                                f"[CRITICAL] GPU {gpu_idx} — Silent Data Corruption detected. "
                                f"matmul validation hash mismatch (seed={seed}). "
                                f"GPU {gpu_idx} hash differs from fleet majority. "
                                f"Training results on this GPU may be invalid. "
                                f"Evacuate workloads and run extended diagnostics."
                            ),
                            context = {
                                "severity":       "critical",
                                "sdc_detected":   True,
                                "seed":           seed,
                                "gpu_hash":       h,
                                "expected_hash":  majority_hash,
                                "affected_gpus":  [gpu_idx],
                                "sdc_count":      self._sdc_count[gpu_idx],
                            },
                        ))

        # Establish reference hashes for single-GPU deployments
        elif len(valid_results) == 1:
            r = valid_results[0]
            if seed not in self._reference_hashes:
                self._reference_hashes[seed] = {}
            prev = self._reference_hashes[seed].get(r.gpu_index)
            if prev and prev != r.actual_hash:
                self._sdc_count[r.gpu_index] = self._sdc_count.get(r.gpu_index, 0) + 1
                alerts.append(AlertEvent(
                    gpu_index       = r.gpu_index,
                    timestamp       = timestamp,
                    state           = GPUState.CRITICAL,
                    prev_state      = GPUState.CLEAN_IDLE,
                    rtheta          = None,
                    rtheta_baseline = None,
                    drift_sigma     = None,
                    confidence      = 0.80,
                    message         = (
                        f"[CRITICAL] GPU {r.gpu_index} — Silent Data Corruption suspected. "
                        f"Validation hash changed between runs (seed={seed}). "
                        f"Single-GPU detection — confidence lower than multi-GPU cross-check. "
                        f"Verify with a second GPU if possible."
                    ),
                    context = {
                        "severity":     "critical",
                        "sdc_detected": True,
                        "seed":         seed,
                        "prev_hash":    prev,
                        "curr_hash":    r.actual_hash,
                    },
                ))
            else:
                self._reference_hashes[seed][r.gpu_index] = r.actual_hash
                log.debug("SDC check passed gpu=%d seed=%d", r.gpu_index, seed)

        return alerts

    def total_sdc_events(self) -> int:
        return sum(self._sdc_count.values())
