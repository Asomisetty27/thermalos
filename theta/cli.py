"""
Theta CLI — theta <command>

Commands:
  setup       Interactive setup wizard (run this first)
  monitor     Run the monitoring agent (blocks)
  baseline    Run a baseline-only idle window scan
  calibrate   Measure hardware-specific R_theta thresholds (run once on non-T4 GPUs)
  classify    Single-snapshot classify all GPUs
  serve       Run agent + Prometheus metrics server only
  train       Retrain bundled models from Stage 1 CSV
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from . import __version__

app     = typer.Typer(
    name="theta",
    add_completion=False,
    pretty_exceptions_enable=False,
    help="GPU thermal-power forensics. Run [bold green]theta setup[/] to get started.",
)
console = Console()


# ── Saved-config helpers ──────────────────────────────────────────────────────

_CONFIG_PATH = Path.home() / ".theta" / "config.json"


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
    """Run the Theta monitoring agent. Reads ~/.theta/config.json if present (CLI flags override)."""
    from .agent.daemon import ThetaAgent, AgentConfig

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
        use_redfish       = saved.get("use_redfish", False),
        redfish_host      = saved.get("redfish_host"),
        redfish_user      = saved.get("redfish_user"),
        redfish_password  = saved.get("redfish_password"),
    )

    # rebind so the banner below uses resolved values
    interval, port = interval_v, port_v

    agent = ThetaAgent(cfg)

    console.print(
        f"[bold green]Theta v{__version__}[/bold green]  "
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


# ── calibrate ─────────────────────────────────────────────────────────────────

@app.command()
def calibrate(
    gpu:              int            = typer.Option(0,     "--gpu",              "-g", help="GPU index to calibrate"),
    idle_wait:        float          = typer.Option(120.0, "--idle-wait",              help="Max seconds to wait for stable idle"),
    load_wait:        float          = typer.Option(120.0, "--load-wait",              help="Max seconds to wait for stable load"),
    skip_load:        bool           = typer.Option(False, "--skip-load",              help="Calibrate idle only (skip load phase)"),
    ambient:          Optional[float]= typer.Option(None,  "--ambient",           "-a", help="Known ambient/coolant temp °C — skips idle-phase wait entirely (use for always-busy DGX nodes)"),
    calibration_file: Optional[str]  = typer.Option(None,  "--calibration-file",        help="Write calibration to this path instead of ~/.theta/calibration.json (use for shared service installs)"),
):
    """
    Measure hardware-specific R_theta thresholds and save to ~/.theta/calibration.json.

    Run this once after setup on any GPU that is not a Tesla T4. The bundled
    classifiers are trained on T4 Stage 1 data — they will misclassify on
    hardware with a different thermal envelope (A100, H100, B200, etc.).

    Two phases:
      1. Idle phase   — waits for the GPU to reach stable idle automatically.
                        Skip with --ambient <temp> if the GPU is never truly idle
                        (e.g., always-running DGX nodes at an AI Factory).
      2. Load phase   — prompts you to start a workload, then locks the load R_theta.

    For shared production installs (systemd service user):
      theta calibrate --gpu 0 --calibration-file /etc/theta/calibration.json
    """
    import asyncio
    from pathlib import Path as _Path
    from .agent.baseline import BaselineManager
    from .agent.collector import NVMLCollector, CollectorConfig
    from .agent.calibrate import (
        CalibrationManager, CalibrationResult,
        derive_thresholds, run_idle_phase, run_load_phase,
    )

    cal_file = _Path(calibration_file) if calibration_file else None
    cal_mgr = CalibrationManager(_file=cal_file)
    existing = cal_mgr.get(gpu)
    if existing:
        console.print(
            f"[dim]Existing calibration for GPU {gpu}: "
            f"load_threshold={existing.load_threshold} C/W  "
            f"idle_threshold={existing.idle_threshold} C/W  "
            f"age={existing.age_hours():.1f}h[/dim]"
        )

    # ── Get GPU name from NVML ────────────────────────────────────────────────
    gpu_name = f"GPU {gpu}"
    try:
        import pynvml
        pynvml.nvmlInit()
        handle   = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        gpu_name = pynvml.nvmlDeviceGetName(handle).decode() if isinstance(
            pynvml.nvmlDeviceGetName(handle), bytes
        ) else pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()
    except Exception:
        pass

    console.print(f"\n[bold]theta calibrate[/bold] — {gpu_name} (GPU {gpu})\n")

    bm = BaselineManager()

    # ── Phase 1: Idle (or --ambient bypass) ───────────────────────────────────
    rtheta_idle: Optional[float] = None

    if ambient is not None:
        # Bypass idle-phase wait when the ambient temperature is known externally
        # (BMC reading, coolant inlet sensor) or when the GPU is always busy.
        # We seed T_ref from the supplied ambient and compute R_θ_idle from the
        # hardware profile's expected idle R_θ value.
        from .agent.hw_profiles import resolve_or_default
        profile = resolve_or_default(gpu_name)
        bm.set_external_ambient(gpu, ambient, source="manual_ambient_flag")

        # Derive expected idle R_θ from profile + supplied ambient.
        # R_θ_idle = profile.rtheta_expected_idle (physics-based estimate).
        # This gives the classifier a plausible idle threshold even without
        # observing an actual idle window — better than T4 defaults on B200/H100.
        rtheta_idle = profile.rtheta_expected_idle
        console.print(
            f"[yellow]⚠[/yellow]  [bold]--ambient mode[/bold]: skipping idle-phase wait.\n"
            f"  Using supplied ambient [bold]{ambient:.1f} °C[/bold] as T_ref, "
            f"profile R_θ_idle estimate [bold]{rtheta_idle:.4f} C/W[/bold].\n"
            f"  [dim]Accuracy is lower than an observed idle window. "
            f"Re-run without --ambient during a maintenance window for best results.[/dim]\n"
        )
    else:
        console.print("[bold cyan]Phase 1 — Idle[/bold cyan]")
        console.print(f"  Waiting up to {idle_wait:.0f}s for stable idle (util < 5%)…")
        console.print(
            "  [dim]Make sure no compute workloads are running.\n"
            "  On always-busy nodes (DGX, AI Factory), use: "
            "[bold]theta calibrate --ambient <inlet_temp_c>[/bold][/dim]\n"
        )

        async def _idle():
            cfg = CollectorConfig(interval_sec=2.0, gpu_indices=[gpu])
            async with NVMLCollector(cfg) as c:
                return await run_idle_phase(c, bm, max_wait_sec=idle_wait)

        rtheta_idle = asyncio.run(_idle())

        if rtheta_idle is None:
            console.print(
                "[red]✗[/red] Idle phase timed out.\n"
                "  Options:\n"
                "  • Free the GPU and retry\n"
                f"  • Use [bold]theta calibrate --gpu {gpu} --ambient <inlet_temp_c>[/bold] "
                "to bypass with a known temperature\n"
                f"  • Use [bold]theta baseline --gpu {gpu} --manual <T>[/bold] to set T_ref manually"
            )
            raise typer.Exit(code=1)

        console.print(f"[green]✓[/green] Idle R_θ locked: [bold]{rtheta_idle:.4f} C/W[/bold]\n")

    # ── Phase 2: Load (optional) ──────────────────────────────────────────────
    rtheta_load: Optional[float] = None

    if not skip_load:
        console.print("[bold cyan]Phase 2 — Load[/bold cyan]")
        console.print("  Start a GPU compute workload now (training job, inference loop, stress test).")
        console.print("  Press [bold]Enter[/bold] when the workload is running…", end=" ")
        try:
            input()
        except EOFError:
            pass

        console.print(f"  Waiting up to {load_wait:.0f}s for stable load (util > 70%)…\n")

        async def _load():
            cfg = CollectorConfig(interval_sec=2.0, gpu_indices=[gpu])
            async with NVMLCollector(cfg) as c:
                return await run_load_phase(c, bm, max_wait_sec=load_wait)

        rtheta_load = asyncio.run(_load())

        if rtheta_load is None:
            console.print(
                "[yellow]⚠[/yellow] Load phase timed out — GPU utilization did not reach 70%. "
                "Calibrating with idle phase only."
            )
        else:
            console.print(f"[green]✓[/green] Load R_θ locked: [bold]{rtheta_load:.4f} C/W[/bold]\n")

    # ── Derive thresholds + save ──────────────────────────────────────────────
    load_threshold, idle_threshold = derive_thresholds(rtheta_idle, rtheta_load)
    source = "observed_both" if rtheta_load is not None else "idle_only"

    result = CalibrationResult(
        gpu_index       = gpu,
        gpu_name        = gpu_name,
        rtheta_idle     = rtheta_idle,
        rtheta_load     = rtheta_load,
        load_threshold  = load_threshold,
        idle_threshold  = idle_threshold,
        calibrated_at   = time.time(),
        source          = source,
    )
    cal_mgr.set(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    from rich.table import Table
    from rich import box

    t = Table(box=box.SIMPLE_HEAVY, show_header=True)
    t.add_column("Metric",    style="bold")
    t.add_column("Value",     justify="right")
    t.add_column("Notes")

    t.add_row("GPU",              gpu_name,                         "")
    t.add_row("R_θ idle",         f"{rtheta_idle:.4f} C/W",         "stable 20s window")
    t.add_row("R_θ load",
              f"{rtheta_load:.4f} C/W" if rtheta_load else "—",
              "stable 20s window" if rtheta_load else "not observed")
    t.add_row("load_threshold",   f"{load_threshold:.3f} C/W",      "R_θ ≤ this → under_load")
    t.add_row("idle_threshold",   f"{idle_threshold:.3f} C/W",      "R_θ ≥ this → idle territory")
    t.add_row("source",           source,                           "")
    t.add_row("saved to",         str(cal_mgr._file),               "")

    console.print(t)
    console.print("\n[green]Calibration complete.[/green] "
                  "Run [bold]theta monitor[/bold] to start using calibrated thresholds.")


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
    from .agent.daemon import ThetaAgent, AgentConfig
    cfg   = AgentConfig(interval_sec=interval, prometheus_port=port, quiet=True)
    agent = ThetaAgent(cfg)
    console.print(f"[green]Metrics:[/green] http://localhost:{port}/metrics")
    asyncio.run(agent.run())


if __name__ == "__main__":
    app()
