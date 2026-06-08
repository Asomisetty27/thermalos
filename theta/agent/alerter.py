"""
Alert routing: stdout (rich), webhook (Slack/PagerDuty/generic), JSONL file.

All alerters are async. The WebhookAlerter uses httpx with retry logic.
Alert payload includes full context (previous state, R_theta history, drift σ).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .metrics import AlertEvent, STATE_LABELS
from .. import __version__

log = logging.getLogger(__name__)

SEVERITY_EMOJI = {
    "critical": "🔴",
    "warning":  "🟡",
    "info":     "🟢",
}


def _severity(event: AlertEvent) -> str:
    ctx = event.context
    return ctx.get("severity", "info") if isinstance(ctx, dict) else "info"


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseAlerter(ABC):
    @abstractmethod
    async def send(self, event: AlertEvent) -> None: ...

    async def close(self) -> None: ...


# ── Stdout ────────────────────────────────────────────────────────────────────

class StdoutAlerter(BaseAlerter):
    """Human-readable, color-coded alerts to stdout via rich."""

    def __init__(self, use_rich: bool = True):
        self._rich = use_rich
        try:
            from rich.console import Console
            self._console = Console()
        except ImportError:
            self._rich = False

    async def send(self, event: AlertEvent) -> None:
        sev   = _severity(event)
        ts    = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).strftime("%H:%M:%S")
        state = STATE_LABELS.get(event.state, event.state.name)
        r_str = f"R_θ={event.rtheta:.4f}" if event.rtheta else "R_θ=n/a"

        color_map = {"critical": "red", "warning": "yellow", "info": "green"}
        color     = color_map.get(sev, "white")

        if self._rich:
            from rich.panel import Panel
            body  = f"[bold]{event.message}[/bold]\n"
            body += f"[dim]{r_str}  ·  conf={event.confidence:.2f}"
            if event.drift_sigma:
                body += f"  ·  drift={event.drift_sigma:.1f}σ"
            body += "[/dim]"
            self._console.print(Panel(body, title=f"[{color}]GPU {event.gpu_index} · {ts}[/{color}]", border_style=color))
        else:
            print(f"[{ts}] GPU {event.gpu_index} [{sev.upper()}] {state} — {r_str}  {event.message}")


# ── Webhook ───────────────────────────────────────────────────────────────────

class WebhookAlerter(BaseAlerter):
    """
    POST JSON alert to a webhook URL (Slack, PagerDuty, generic HTTP).

    Slack format: attach a blocks payload if url contains "slack".
    Generic format: flat JSON with full context.

    Retries up to 3 times with exponential backoff (SSL/network resilience —
    same pattern as theta_e003_rerun.py sheet writes).
    """

    def __init__(self, url: str, max_retries: int = 3):
        self._url         = url
        self._max_retries = max_retries
        self._client      = None
        self._is_slack    = "slack" in url

    async def _get_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    def _build_payload(self, event: AlertEvent) -> dict:
        sev   = _severity(event)
        state = STATE_LABELS.get(event.state, event.state.name)
        ts    = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat()

        if self._is_slack:
            color_map = {"critical": "#B83030", "warning": "#C8942A", "info": "#27A05A"}
            return {
                "attachments": [{
                    "color": color_map.get(sev, "#888888"),
                    "fallback": event.message,
                    "fields": [
                        {"title": "GPU",         "value": str(event.gpu_index),                                  "short": True},
                        {"title": "State",       "value": state,                                                  "short": True},
                        {"title": "R_θ",         "value": f"{event.rtheta:.4f} C/W" if event.rtheta else "n/a",  "short": True},
                        {"title": "Drift",       "value": f"{event.drift_sigma:.1f}σ" if event.drift_sigma else "n/a", "short": True},
                        {"title": "Confidence",  "value": f"{event.confidence:.2f}",                             "short": True},
                        {"title": "Timestamp",   "value": ts,                                                     "short": True},
                    ],
                    "text":   event.message,
                    "footer": "Theta",
                    "ts":     int(event.timestamp),
                }]
            }

        return {
            "source":           "theta",
            "version":          __version__,
            "timestamp":        ts,
            "severity":         sev,
            "gpu_index":        event.gpu_index,
            "state":            state,
            "prev_state":       STATE_LABELS.get(event.prev_state, "unknown"),
            "rtheta":           event.rtheta,
            "rtheta_baseline":  event.rtheta_baseline,
            "drift_sigma":      event.drift_sigma,
            "confidence":       event.confidence,
            "message":          event.message,
            "context":          event.context,
        }

    async def send(self, event: AlertEvent) -> None:
        payload = self._build_payload(event)
        client  = await self._get_client()

        for attempt in range(self._max_retries):
            try:
                r = await client.post(self._url, json=payload)
                r.raise_for_status()
                return
            except Exception as e:
                wait = 3 ** attempt
                log.warning(f"Webhook attempt {attempt + 1} failed: {e}. Retrying in {wait}s")
                await asyncio.sleep(wait)

        log.error(f"Webhook failed after {self._max_retries} attempts for GPU {event.gpu_index}")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()


# ── File (JSONL) ──────────────────────────────────────────────────────────────

class FileAlerter(BaseAlerter):
    """Append-only JSONL alert log. One JSON object per line."""

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def send(self, event: AlertEvent) -> None:
        record = {
            "version":   __version__,
            "ts":        event.timestamp,
            "gpu":       event.gpu_index,
            "severity":  _severity(event),
            "state":     STATE_LABELS.get(event.state, "unknown"),
            "prev":      STATE_LABELS.get(event.prev_state, "unknown"),
            "rtheta":    event.rtheta,
            "baseline":  event.rtheta_baseline,
            "sigma":     event.drift_sigma,
            "conf":      event.confidence,
            "message":   event.message,
        }
        line = json.dumps(record)
        # Async append
        await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        with open(self._path, "a") as f:
            f.write(line + "\n")


# ── Deduplication ─────────────────────────────────────────────────────────────

class _AlertDeduper:
    """Sliding-window deduplication keyed by (gpu_index, type, severity).

    Audit finding addressed: a GPU oscillating UNDER_LOAD ↔ DRIFTING in 5s
    ticks generated one webhook POST per transition — five per minute into
    Slack. The router had no concept of "we just sent this exact alert."

    Now: emit ONCE per unique (gpu_index, alert_type, severity) tuple per
    `cooldown_sec` window. Subsequent duplicates are silently dropped and a
    suppression count is attached to the eventual next emission.

    Different alert types (drift vs ECC) on the same GPU are NOT deduped —
    they're independent failure modes that may need independent action.
    """

    def __init__(self, cooldown_sec: float = 60.0):
        self._cooldown_sec = cooldown_sec
        # key → (last_emit_ts, suppressed_count_since_last_emit)
        self._last_emit: dict[tuple, tuple[float, int]] = {}

    @staticmethod
    def _key(event: AlertEvent) -> tuple:
        # Use the *severity tier* not the raw rtheta value — a drift alert at
        # 2.5σ and 2.6σ should dedupe together, but a critical at 3.5σ should
        # punch through a previous warning at 2.0σ.
        return (
            event.gpu_index,
            event.alert_type if hasattr(event, "alert_type") else "unknown",
            event.severity if hasattr(event, "severity") else "info",
        )

    def should_emit(self, event: AlertEvent, now: Optional[float] = None) -> tuple[bool, int]:
        """Return (should_send, suppressed_count_so_far).

        Caller can attach `suppressed_count` to the outgoing payload so
        operators see "fired 8 times in the last minute" rather than getting
        a single alert with no context.
        """
        t = now if now is not None else time.time()
        k = self._key(event)
        last, suppressed = self._last_emit.get(k, (0.0, 0))
        if t - last < self._cooldown_sec:
            self._last_emit[k] = (last, suppressed + 1)
            return False, suppressed + 1
        # Emit, reset suppression counter
        prior_suppressed = suppressed
        self._last_emit[k] = (t, 0)
        return True, prior_suppressed


# ── Router ────────────────────────────────────────────────────────────────────

class AlertRouter:
    """
    Fan-out: send every AlertEvent to all registered alerters concurrently.

    Now includes a deduplication layer to prevent alert storms when a GPU
    oscillates near a threshold. Pass `cooldown_sec=0` to disable.
    """

    def __init__(
        self,
        alerters: list[BaseAlerter] | None = None,
        cooldown_sec: float = 60.0,
    ):
        self._alerters: list[BaseAlerter] = alerters or []
        self._deduper = _AlertDeduper(cooldown_sec) if cooldown_sec > 0 else None

    def add(self, alerter: BaseAlerter) -> None:
        self._alerters.append(alerter)

    async def route(self, event: AlertEvent) -> None:
        if not self._alerters:
            return
        if self._deduper is not None:
            should, suppressed = self._deduper.should_emit(event)
            if not should:
                return
            # Attach the suppression count so the payload is self-describing.
            # Touch only the context dict — don't mutate the event identity.
            try:
                if event.context is None:
                    event.context = {}
                if suppressed > 0:
                    event.context["suppressed_duplicates"] = suppressed
            except AttributeError:
                pass
        await asyncio.gather(
            *(a.send(event) for a in self._alerters),
            return_exceptions=True,
        )

    async def close(self) -> None:
        await asyncio.gather(*(a.close() for a in self._alerters), return_exceptions=True)
