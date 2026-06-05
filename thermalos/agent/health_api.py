"""
ThermalOS Health API — /api/v1/health

Lightweight HTTP server (stdlib only, no new deps) exposing a single health
score per GPU. Designed for two callers:

  SLURM prolog:  curl -s http://localhost:9102/api/v1/health | jq '.gpu_0.risk'
  MPI runtime:   poll /api/v1/health/gpu/0 before scheduling a replica

Runs in a daemon thread alongside the Prometheus exporter. Default port 9102.

Response shape:
  GET /api/v1/health
  {
    "agent_version": "0.1.8",
    "uptime_ticks": 1200,
    "gpus": {
      "0": {
        "state": "clean_idle",
        "score": 0.94,       # 0–1, higher = healthier
        "risk": 0.06,        # degradation risk 0–1
        "recommendation": "ok",  # ok | watch | drain | evacuate
        "rtheta": 1.21,
        "t_ref": 36.5,
        "baseline_locked": true,
        "poll_latency_ms": 2.1
      }
    }
  }

  GET /api/v1/health/gpu/0  →  same as gpus["0"]
  GET /api/v1/ready          →  200 {"ready": true} or 503
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Callable, Optional

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def _recommendation(state: str, risk: float) -> str:
    if state in ("critical", "zombie_recovery"):
        return "evacuate"
    if risk >= 0.80 or state == "drifting":
        return "drain"
    if risk >= 0.50:
        return "watch"
    return "ok"


class HealthRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for health API requests."""

    def __init__(self, get_status: Callable, get_poll_latency: Callable, *args, **kwargs):
        self._get_status       = get_status
        self._get_poll_latency = get_poll_latency
        super().__init__(*args, **kwargs)

    def _json(self, data: dict, code: int = 200) -> None:
        body = json.dumps(data, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
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

    def log_message(self, fmt, *args) -> None:
        log.debug("health_api " + fmt, *args)


class HealthAPIServer:
    """
    Threaded HTTP server for the health API.
    Shares state with the daemon via callbacks (no shared mutable objects).
    """

    def __init__(
        self,
        port:             int,
        get_status:       Callable,
        get_poll_latency: Callable,
    ):
        self._port            = port
        self._get_status      = get_status
        self._get_poll_latency= get_poll_latency
        self._server:  Optional[HTTPServer] = None
        self._thread:  Optional[threading.Thread] = None

    def start(self) -> None:
        get_status       = self._get_status
        get_poll_latency = self._get_poll_latency

        def handler_factory(*args, **kwargs):
            return HealthRequestHandler(get_status, get_poll_latency, *args, **kwargs)

        try:
            self._server = HTTPServer(("0.0.0.0", self._port), handler_factory)
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True, name="thermalos-health-api"
            )
            self._thread.start()
            log.info("health_api_started", port=self._port,
                     url=f"http://localhost:{self._port}/api/v1/health")
        except OSError as e:
            log.warning("health_api_failed_to_start", port=self._port, error=str(e))

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
