"""
Theta Health API — /api/v1/health  + /api/v1/agent/*

Lightweight HTTP server (stdlib only, no new deps) exposing GPU health +
agent state. Designed for three callers:

  SLURM prolog:        curl -sH "Authorization: Bearer $T" \\
                            http://localhost:9102/api/v1/health | jq '.gpu_0.risk'
  MPI runtime:         poll /api/v1/health/gpu/0 before scheduling a replica
  Agent Control Center (site): fetch /api/v1/agent/fleet/status every 5s

Auth: when a bearer token is configured (config.health_api_token, env var
$THETA_HEALTH_TOKEN, or ~/.theta/health.token file), all requests must
include `Authorization: Bearer <token>` or get a 401. When no token is
configured, the API runs in open mode and logs a one-time warning that
operators should configure a token before exposing the port on a shared
network.

Endpoints:
  GET /api/v1/ready                  → 200 if any GPU has a baseline locked
  GET /api/v1/health                 → fleet score summary (SLURM/MPI use)
  GET /api/v1/health/gpu/{i}         → single-GPU summary
  GET /api/v1/agent/fleet/status     → rich snapshot for the site's UI
  GET /api/v1/agent/gpu/{i}/details  → per-GPU causal explanation +
                                       maintenance score + decision log
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)


def _recommendation(state: str, risk: float) -> str:
    if state in ("critical", "zombie_recovery"):
        return "evacuate"
    if risk >= 0.80 or state == "drifting":
        return "drain"
    if risk >= 0.50:
        return "watch"
    return "ok"


def _resolve_token(explicit: Optional[str] = None) -> Optional[str]:
    """Find the bearer token from (in priority order): explicit arg, env var,
    on-disk file. Returns None when nothing is configured."""
    if explicit:
        return explicit
    env = os.environ.get("THETA_HEALTH_TOKEN")
    if env:
        return env.strip()
    token_file = Path.home() / ".theta" / "health.token"
    if token_file.exists():
        try:
            return token_file.read_text().strip()
        except OSError:
            return None
    return None


class HealthRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for health API requests."""

    # Class-level so the factory closure doesn't need to thread it through
    # every constructor argument list.
    auth_token: Optional[str] = None

    def __init__(
        self,
        get_status: Callable,
        get_poll_latency: Callable,
        get_agent_details: Optional[Callable],
        *args,
        **kwargs,
    ):
        self._get_status        = get_status
        self._get_poll_latency  = get_poll_latency
        self._get_agent_details = get_agent_details
        super().__init__(*args, **kwargs)

    def _json(self, data: dict, code: int = 200) -> None:
        body = json.dumps(data, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # CORS — the Agent Control Center UI fetches from a different origin
        # in dev. In production deployments, restrict via a reverse proxy.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        """Return True if request is authorized (or auth disabled)."""
        if not self.auth_token:
            return True  # open mode — warned at startup
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return False
        # Constant-time comparison to defeat timing attacks
        import hmac
        provided = header[len("Bearer "):].strip()
        return hmac.compare_digest(provided, self.auth_token)

    def do_OPTIONS(self) -> None:  # noqa: N802 — CORS preflight
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        # Auth gate before any work — even readiness probes go through it.
        # If you want unauthenticated readiness, run agent with no token
        # configured (and accept the network-trust assumption).
        if not self._check_auth():
            self._json({"error": "unauthorized"}, 401)
            return

        path = self.path.rstrip("/")

        if path == "/api/v1/ready":
            status = self._get_status()
            ready  = len(status.get("gpus", {})) > 0
            self._json({"ready": ready}, 200 if ready else 503)

        elif path == "/api/v1/health":
            self._json(self._build_fleet_health())

        elif path.startswith("/api/v1/health/gpu/"):
            idx_str = path.split("/")[-1]
            try:
                idx = int(idx_str)
            except ValueError:
                self._json({"error": "invalid gpu index"}, 400)
                return
            fleet = self._build_fleet_health()
            gpu_data = fleet["gpus"].get(str(idx))
            if gpu_data is None:
                self._json({"error": f"gpu {idx} not found"}, 404)
            else:
                self._json(gpu_data)

        elif path == "/api/v1/agent/fleet/status":
            self._json(self._build_fleet_status())

        elif path.startswith("/api/v1/agent/gpu/") and path.endswith("/details"):
            try:
                idx = int(path.split("/")[-2])
            except (ValueError, IndexError):
                self._json({"error": "invalid gpu index"}, 400)
                return
            if self._get_agent_details is None:
                self._json({"error": "agent details endpoint not wired"}, 501)
                return
            details = self._get_agent_details(idx)
            if details is None:
                self._json({"error": f"gpu {idx} not found"}, 404)
            else:
                self._json(details)

        else:
            self._json({"error": "not found"}, 404)

    def _build_fleet_health(self) -> dict:
        status       = self._get_status()
        poll_latency = self._get_poll_latency()

        gpus_out = {}
        for idx_str, gpu in status.get("gpus", {}).items():
            state      = gpu.get("state", "unknown").lower()
            risk       = round(gpu.get("degradation_risk", 0.0), 3)
            score      = round(1.0 - risk, 3)
            rtheta     = gpu.get("rtheta")
            t_ref      = gpu.get("t_ref")
            lat_ms     = round(poll_latency.get(int(idx_str), 0.0) * 1000, 2)

            gpus_out[idx_str] = {
                "state":           state,
                "score":           score,
                "risk":            risk,
                "recommendation":  _recommendation(state, risk),
                "rtheta":          round(rtheta, 4) if rtheta else None,
                "t_ref":           round(t_ref, 2)  if t_ref  else None,
                "baseline_locked": gpu.get("baseline_locked", False),
                "poll_latency_ms": lat_ms,
            }

        return {
            "agent_version":  status.get("agent_version", "unknown"),
            "uptime_ticks":   status.get("uptime_ticks", 0),
            "alerts":         status.get("alerts", 0),
            "gpus":           gpus_out,
        }

    def _build_fleet_status(self) -> dict:
        """Richer snapshot for the Agent Control Center UI.

        Layered on top of the basic health response: adds smoothed state +
        confidence (from temporal filter), causal headline (from causal
        engine), and maintenance priority (from maintenance scorer) when
        those are available via _get_agent_details. Degrades gracefully
        when the daemon hasn't wired them yet — site still sees the
        baseline health fields.
        """
        base = self._build_fleet_health()
        if self._get_agent_details is None:
            return {**base, "agent_capabilities": ["health"], "timestamp": time.time()}

        # Annotate each GPU with its rich detail object's headline fields
        # (the FULL detail object is fetched via the per-GPU endpoint).
        for idx_str in base["gpus"]:
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            extra = self._get_agent_details(idx)
            if not extra:
                continue
            causal = extra.get("causal_explanation")
            maint  = extra.get("maintenance")
            smoothed = extra.get("smoothed_state")
            base["gpus"][idx_str].update({
                "headline":           (causal or {}).get("headline"),
                "urgency":            (causal or {}).get("urgency"),
                "smoothed_state":     (smoothed or {}).get("state"),
                "smoothed_confidence": (smoothed or {}).get("confidence"),
                "maintenance_priority": (maint or {}).get("priority"),
                "days_until_service":  (maint or {}).get("days_until_service"),
            })

        base["agent_capabilities"] = ["health", "reasoning", "memory", "adaptability"]
        base["timestamp"] = time.time()
        return base

    def log_message(self, fmt, *args) -> None:
        log.debug("health_api " + fmt, *args)


class HealthAPIServer:
    """
    Threaded HTTP server for the health API.
    Shares state with the daemon via callbacks (no shared mutable objects).
    """

    def __init__(
        self,
        port:              int,
        get_status:        Callable,
        get_poll_latency:  Callable,
        get_agent_details: Optional[Callable] = None,
        auth_token:        Optional[str] = None,
        bind_host:         str = "0.0.0.0",
    ):
        self._port              = port
        self._get_status        = get_status
        self._get_poll_latency  = get_poll_latency
        self._get_agent_details = get_agent_details
        self._bind_host         = bind_host
        self._auth_token        = _resolve_token(auth_token)
        self._server:  Optional[HTTPServer] = None
        self._thread:  Optional[threading.Thread] = None

    def start(self) -> None:
        get_status        = self._get_status
        get_poll_latency  = self._get_poll_latency
        get_agent_details = self._get_agent_details
        token             = self._auth_token

        if token is None:
            log.warning(
                "health_api_no_auth — running open. To require bearer-token auth, "
                "set THETA_HEALTH_TOKEN env var, write ~/.theta/health.token, or "
                "pass auth_token to HealthAPIServer. Do not expose this port on "
                "untrusted networks without authentication."
            )

        # Class-level so all handler instances see the same token without
        # threading it through __init__.
        HealthRequestHandler.auth_token = token

        def handler_factory(*args, **kwargs):
            return HealthRequestHandler(
                get_status, get_poll_latency, get_agent_details, *args, **kwargs,
            )

        try:
            self._server = HTTPServer((self._bind_host, self._port), handler_factory)
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True, name="theta-health-api"
            )
            self._thread.start()
            log.info("health_api_started", port=self._port,
                     auth="bearer" if token else "open",
                     url=f"http://localhost:{self._port}/api/v1/health")
        except OSError as e:
            log.warning("health_api_failed_to_start", port=self._port, error=str(e))

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
