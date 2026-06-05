"""
ThermalOS CLI — thermalos <command>

Commands:
  setup       Interactive setup wizard (run this first)
  monitor     Run the monitoring agent (blocks)
  baseline    Run a baseline-only idle window scan
  classify    Single-snapshot classify all GPUs
  serve       Run agent + Prometheus metrics server only
  train       Retrain bundled models from Stage 1 CSV
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from . import __version__

app     = typer.Typer(
    name="thermalos",
    add_completion=False,
    pretty_exceptions_enable=False,
    help="GPU thermal-power forensics. Run [bold green]thermalos setup[/] to get started.",
)
console = Console()


# ── Saved-config helpers ──────────────────────────────────────────────────────

_CONFIG_PATH = Path.home() / ".thermalos" / "config.json"


def _saved_config() -> dict:
    """Load the wizard-saved config if present, else return an empty dict."""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _coalesce(cli_value, config_key: str, saved: dict, default):
    """CLI value (if set) > saved config > default. Treat sentinels as 'unset'."""
    if cli_value is not None and cli_value != "__default__":
        return cli_value
    return saved.get(config_key, default)


# ── setup ─────────────────────────────────────────────────────────────────────

@app.command()
def setup():
    """Interactive setup wizard. Run this first. (~90 seconds)"""
    from .wizard import run_wizard
    run_wizard()


# ── monitor ───────────────────────────────────────────────────────────────────

@app.command()
def monitor(
    interval:    Optional[float] = typer.Option(None, "--interval",   "-i",  help="Sample interval seconds [default: 5.0 or saved config]"),
    gpus:        Optional[str]   = typer.Option(None, "--gpus",       "-g",  help="GPU indices (comma-sep) or 'all' [default: saved config or all]"),
    webhook:     Optional[str]   = typer.Option(None, "--webhook",    "-w",  help="Alert webhook URL [default: saved config or none]"),
    log_file:    Optional[str]   = typer.Option(None, "--log",                help="JSONL alert log file [default: saved config or none]"),
    port:        Optional[int]   = typer.Option(None, "--port",       "-p",  help="Prometheus port; 0 disables [default: saved config or 9101]"),
    quiet:       bool            = typer.Option(False, "--quiet",     "-q",  help="Suppress stdout alerts"),
    dt:          bool            = typer.Option(True,  "--dt/--nb",          help="Use Decision Tree (--dt) or Naive Bayes (--nb)"),
    sigma_warn:  float           = typer.Option(2.0,   "--sigma-warn",       help="Drift warning threshold (σ)"),
    sigma_crit:  float           = typer.Option(3.5,   "--sigma-crit",       help="Drift critical threshold (σ)"),
):
    """Run the ThermalOS monitoring agent. Reads ~/.thermalos/config.json if present (CLI flags override)."""
    from .agent.daemon import ThermalOSAgent, AgentConfig

    saved = _saved_config()

    # Resolve each field: explicit CLI flag > saved config > hardcoded default
    interval_v   = interval if interval is not None else saved.get("interval_sec", 5.0)
    gpus_v       = gpus     if gpus     is not None else (
        ",".join(str(i) for i in saved["gpu_indices"]) if saved.get("gpu_indices") else "all"
    )
    webhook_v    = webhook  if webhook  is not None else saved.get("webhook_url")
    log_file_v   = log_file if log_file is not None else saved.get("alert_log_path")
    port_v       = port     if port     is not None else (
        saved.get("prometheus_port", 9101) if saved.get("enable_prometheus", True) else 0
    )

    gpu_list = None if gpus_v == "all" else [int(g) for g in gpus_v.split(",")]

    cfg = AgentConfig(
        interval_sec      = interval_v,
        gpu_indices       = gpu_list,
        webhook_url       = webhook_v,
        alert_log_path    = log_file_v,
        prometheus_port   = port_v,
        enable_prometheus = port_v > 0,
        quiet             = quiet,
        prefer_dt         = dt,
        k_warn            = sigma_warn,
        k_critical        = sigma_crit,
        data_sharing      = saved.get("data_sharing", False),
    )

    # rebind so the banner below uses resolved values
    interval, port = interval_v, port_v

    agent = ThermalOSAgent(cfg)

    console.print(
        f"[bold green]ThermalOS v{__version__}[/bold green]  "
        f"[dim]interval={interval}s  classifier={agent._classifier.mode}  "
        f"{'prometheus:' + str(port) if port > 0 else 'no-metrics'}[/dim]"
    )

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


# ── baseline ──────────────────────────────────────────────────────────────────

@app.command()
def baseline(
    gpu:      int   = typer.Option(0,    "--gpu",      "-g",  help="GPU index to baseline"),
    duration: float = typer.Option(60.0, "--duration", "-d",  help="Max wait time for stable idle window (sec)"),
    manual:   Optional[float] = typer.Option(None, "--manual", "-m", help="Set T_ref manually (skip idle window)"),
):
    """Estimate or set virtual ambient temperature (T_ref) for a GPU."""
    from .agent.baseline import BaselineManager
    from .agent.collector import NVMLCollector, CollectorConfig

    bm = BaselineManager()

    if manual is not None:
        bm.set_manual(gpu, manual)
        console.print(f"[green]GPU {gpu}[/green] T_ref manually set to [bold]{manual:.1f} °C[/bold]")
        return

    console.print(f"Waiting for stable idle window on GPU {gpu} (up to {duration:.0f}s)…")

    async def _run():
        cfg = CollectorConfig(interval_sec=2.0, gpu_indices=[gpu])
        async with NVMLCollector(cfg) as c:
            deadline = asyncio.get_event_loop().time() + duration
            async for s in c.stream():
                bm.update(gpu, s.temp_junction, s.util_pct, s.perf_state, s.timestamp)
                if bm.has_baseline(gpu):
                    b = bm.get_baseline(gpu)
                    console.print(
                        f"[green]✓[/green] T_ref locked at [bold]{b.t_ref:.2f} °C[/bold]  "
                        f"σ={b.sigma:.3f}  n={b.n_samples}"
                    )
                    return
                if asyncio.get_event_loop().time() > deadline:
                    console.print("[yellow]Timeout — no stable idle window found. "
                                  "Use --manual to set T_ref explicitly.[/yellow]")
                    return
                console.print(f"  [dim]T={s.temp_junction:.1f}°C  util={s.util_pct:.0f}%  "
                               f"P-state=P{s.perf_state}[/dim]", end="\r")

    asyncio.run(_run())


# ── classify ──────────────────────────────────────────────────────────────────

@app.command()
def classify(
    gpus:   str  = typer.Option("all", "--gpus",    "-g", help="GPU indices or 'all'"),
    raw:    bool = typer.Option(False, "--raw",            help="Skip steady-state filter"),
):
    """Snapshot: classify current state of all GPUs."""
    from .agent.collector import NVMLCollector, CollectorConfig
    from .agent.baseline  import BaselineManager
    from .agent.metrics   import enrich
    from .agent.window    import SteadyStateWindow
    from .agent.classifier import StateClassifier
    from .agent.metrics    import STATE_LABELS  # noqa: F401  (used in _print_classify_table)

    gpu_list = None if gpus == "all" else [int(g) for g in gpus.split(",")]

    async def _run():
        cfg  = CollectorConfig(interval_sec=1.0, gpu_indices=gpu_list)
        bm   = BaselineManager()
        win  = SteadyStateWindow(window_sec=15.0)
        clf  = StateClassifier()

        console.print("[dim]Collecting samples for steady-state window (15s)…[/dim]")

        async with NVMLCollector(cfg) as c:
            seen: dict[int, list] = {}
            async for s in c.stream():
                t_ref    = bm.get_t_ref(s.gpu_index)
                enriched = enrich(s, t_ref)
                bm.update(s.gpu_index, s.temp_junction, s.util_pct, s.perf_state, s.timestamp)

                if enriched.rtheta is None:
                    continue

                window = win.update(
                    s.gpu_index, s.timestamp,
                    enriched.rtheta, s.power_w, s.util_pct, s.perf_state
                )

                if window.is_stable or raw:
                    seen[s.gpu_index] = window

                if set(seen.keys()) == set(gpu_list or range(4)):
                    break

            _print_classify_table(seen, clf)

    asyncio.run(_run())


def _print_classify_table(windows, clf) -> None:
    from .agent.metrics import STATE_LABELS

    t = Table(box=box.SIMPLE_HEAVY, show_header=True)
    t.add_column("GPU",       style="bold")
    t.add_column("State",     style="bold")
    t.add_column("R_θ (C/W)", justify="right")
    t.add_column("σ (window)", justify="right")
    t.add_column("Conf",      justify="right")
    t.add_column("Reason")

    state_colors = {
        "under_load":          "green",
        "clean_idle":          "blue",
        "zombie_recovery":     "red",
        "child_exit_recovery": "yellow",
        "drifting":            "yellow",
        "critical":            "red",
        "unknown":             "dim",
    }

    for gpu_idx, window in sorted(windows.items()):
        state, conf = clf.classify(window)
        label  = STATE_LABELS.get(state, state.name)
        color  = state_colors.get(label, "white")
        reason = clf.explain(window).split("—")[-1].strip()[:60]
        t.add_row(
            str(gpu_idx),
            f"[{color}]{label}[/{color}]",
            f"{window.rtheta_mean:.4f}",
            f"{window.rtheta_std:.4f}",
            f"{conf:.2f}",
            f"[dim]{reason}[/dim]",
        )

    console.print(t)


# ── train ─────────────────────────────────────────────────────────────────────

@app.command()
def train(
    csv: str = typer.Argument(..., help="Path to ThermalOS_Measurements_Raw.csv"),
):
    """Retrain bundled classifier models from Stage 1 CSV data."""
    csv_path = Path(csv)
    if not csv_path.exists():
        console.print(f"[red]Error:[/red] CSV not found: {csv}")
        raise typer.Exit(code=1)
    from .models.train import train as do_train
    do_train(csv_path)


# ── serve ─────────────────────────────────────────────────────────────────────

@app.command()
def serve(
    port: int = typer.Option(9101, "--port", "-p", help="Prometheus metrics port"),
    interval: float = typer.Option(5.0, "--interval", "-i"),
):
    """Run agent with Prometheus metrics export only (no stdout alerts)."""
    from .agent.daemon import ThermalOSAgent, AgentConfig
    cfg   = AgentConfig(interval_sec=interval, prometheus_port=port, quiet=True)
    agent = ThermalOSAgent(cfg)
    console.print(f"[green]Metrics:[/green] http://localhost:{port}/metrics")
    asyncio.run(agent.run())


if __name__ == "__main__":
    app()
