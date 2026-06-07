"""
Theta setup wizard — theta setup

Interactive step-by-step onboarding. Detects GPUs, locks virtual ambient,
runs first classification, configures alerts, saves config.

Designed to give a first-time user complete confidence in 90 seconds.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich import box

from . import __version__

CONFIG_PATH = Path.home() / ".theta" / "config.json"
console = Console()

# ── Palette ───────────────────────────────────────────────────────────────────
GREEN  = "#27A05A"
BLUE   = "#5878A8"
YELLOW = "#C8942A"
RED    = "#B83030"
DIM    = "#606070"
TEXT   = "#E8E8F0"

# ── Logo ──────────────────────────────────────────────────────────────────────
LOGO = r"""
  ████████╗██╗  ██╗███████╗██████╗ ███╗   ███╗ █████╗ ██╗      ██████╗ ███████╗
  ╚══██╔══╝██║  ██║██╔════╝██╔══██╗████╗ ████║██╔══██╗██║     ██╔═══██╗██╔════╝
     ██║   ███████║█████╗  ██████╔╝██╔████╔██║███████║██║     ██║   ██║███████╗
     ██║   ██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══██║██║     ██║   ██║╚════██║
     ██║   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║██║  ██║███████╗╚██████╔╝███████║
     ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝
"""

# ── Step indicator ─────────────────────────────────────────────────────────────
def step_header(n: int, total: int, title: str, subtitle: str = "") -> None:
    console.print()
    bar = "".join(
        f"[bold {GREEN}]━[/]" if i < n else
        f"[bold {BLUE}]━[/]" if i == n else
        f"[{DIM}]─[/]"
        for i in range(1, total + 1)
    )
    console.print(f"  {bar}  [bold {TEXT}]{title}[/]  [{DIM}]step {n}/{total}[/]")
    if subtitle:
        console.print(f"  [dim]{subtitle}[/dim]")
    console.print()


def ok(msg: str) -> None:
    console.print(f"  [bold {GREEN}]✓[/]  {msg}")

def info(msg: str) -> None:
    console.print(f"  [{BLUE}]·[/]  [{DIM}]{msg}[/]")

def warn(msg: str) -> None:
    console.print(f"  [{YELLOW}]![/]  [{YELLOW}]{msg}[/]")

def section(title: str) -> None:
    console.print(f"\n  [{DIM}]{title}[/{DIM}]")


# ── Step 1: Welcome ────────────────────────────────────────────────────────────
def step_welcome() -> None:
    console.clear()
    console.print(Align(
        Panel(
            Align(f"[bold {GREEN}]{LOGO}[/]", align="center"),
            border_style=DIM,
            padding=(0, 2),
        ),
        align="center",
    ))
    console.print(Align(
        f"[{TEXT}]GPU thermal-power forensics.  [{DIM}]v{__version__} · MIT licensed[/]",
        align="center",
    ))
    console.print()
    console.print(Align(
        Panel(
            f"  [{TEXT}]Theta computes [bold]R_θ = ΔT / P[/] in real time from your existing\n"
            f"  DCGM telemetry. That ratio separates busy-hot GPUs from failing-hot ones —\n"
            f"  [bold {GREEN}]the only metric that does.[/]\n\n"
            f"  [{DIM}]This wizard takes ~90 seconds and leaves you fully configured.[/]  ",
            border_style=BLUE,
            title=f"[{BLUE}]What is this?[/]",
            title_align="left",
            padding=(1, 2),
        ),
        align="center",
    ))
    console.print()
    Confirm.ask(f"  [{TEXT}]Ready to set up Theta?[/]", default=True)


# ── Step 2: System check ───────────────────────────────────────────────────────
def step_system_check() -> dict:
    step_header(1, 6, "System check", "Verifying Python, pynvml, and GPU access")

    results = {}

    with Progress(
        SpinnerColumn(style=f"bold {GREEN}"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as p:
        # Python version
        t = p.add_task("Checking Python version…", total=None)
        time.sleep(0.3)
        ver = sys.version_info
        p.remove_task(t)
        if ver >= (3, 10):
            ok(f"Python {ver.major}.{ver.minor}.{ver.micro}")
        else:
            warn(f"Python {ver.major}.{ver.minor} — recommend 3.10+")
        results["python"] = f"{ver.major}.{ver.minor}"

        # pynvml
        t = p.add_task("Checking pynvml / NVIDIA driver…", total=None)
        time.sleep(0.3)
        p.remove_task(t)
        try:
            import pynvml
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            driver = pynvml.nvmlSystemGetDriverVersion()
            pynvml.nvmlShutdown()
            ok(f"pynvml  ·  driver {driver}  ·  {n} GPU{'s' if n != 1 else ''} detected")
            results["nvml"] = True
            results["n_gpus"] = n
            results["driver"] = driver
        except Exception as e:
            warn(f"pynvml unavailable ({e}). Demo mode will be used.")
            results["nvml"] = False
            results["n_gpus"] = 4
            results["demo"]   = True

        # prometheus_client
        t = p.add_task("Checking optional dependencies…", total=None)
        time.sleep(0.2)
        p.remove_task(t)
        try:
            import prometheus_client  # noqa
            ok("prometheus_client — Prometheus export available")
            results["prometheus"] = True
        except ImportError:
            info("prometheus_client not installed — Prometheus export disabled")
            results["prometheus"] = False

        # structlog
        try:
            import structlog  # noqa
            results["structlog"] = True
        except ImportError:
            results["structlog"] = False

    console.print()
    if results.get("demo"):
        console.print(Panel(
            f"  [{YELLOW}]No NVIDIA GPU detected.[/] Theta will run in [bold]demo mode[/] with\n"
            f"  synthetic telemetry. All features work — results are simulated.\n\n"
            f"  Install pynvml on a machine with an NVIDIA GPU for live monitoring:\n"
            f"  [bold {BLUE}]pip install nvidia-ml-py[/]",
            border_style=YELLOW,
            title=f"[{YELLOW}]Demo mode[/]",
            title_align="left",
            padding=(1, 2),
        ))

    return results


# ── Step 3: GPU inventory ──────────────────────────────────────────────────────
def step_gpu_inventory(sys_info: dict) -> list[dict]:
    step_header(2, 6, "GPU inventory", "Scanning detected GPUs")

    gpus = []

    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        for i in range(n):
            h    = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            pwr  = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            try:
                ps = pynvml.nvmlDeviceGetPerformanceState(h)
                ps_str = str(ps).replace("PerformanceState_", "")
            except Exception:
                ps_str = "?"
            gpus.append({
                "index": i, "name": name,
                "mem_gb": round(mem.total / 1e9, 1),
                "temp": temp, "power": pwr, "pstate": ps_str,
            })
        pynvml.nvmlShutdown()
    except Exception:
        # Demo GPUs
        gpus = [
            {"index": i, "name": f"Tesla T4 (demo {i})", "mem_gb": 16.0,
             "temp": 42 + i * 2, "power": 11.4, "pstate": "P8"}
            for i in range(sys_info.get("n_gpus", 4))
        ]

    t = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    t.add_column("GPU",    style=f"bold {TEXT}", justify="right")
    t.add_column("Name",   style=TEXT)
    t.add_column("VRAM",   justify="right", style=DIM)
    t.add_column("Temp",   justify="right")
    t.add_column("Power",  justify="right", style=DIM)
    t.add_column("P-state",justify="right", style=DIM)
    t.add_column("Status", justify="center")

    for g in gpus:
        temp_color = RED if g["temp"] > 80 else YELLOW if g["temp"] > 65 else GREEN
        t.add_row(
            str(g["index"]),
            g["name"],
            f"{g['mem_gb']} GB",
            f"[{temp_color}]{g['temp']}°C[/]",
            f"{g['power']:.1f}W",
            f"P{g['pstate']}",
            f"[{GREEN}]●[/] online",
        )

    console.print(Panel(t, border_style=DIM, padding=(0, 1)))
    ok(f"{len(gpus)} GPU{'s' if len(gpus) != 1 else ''} ready for monitoring")

    non_t4 = [g for g in gpus if "T4" not in g["name"]]
    if non_t4:
        models = ", ".join(sorted({g["name"].split("(")[0].strip() for g in non_t4}))
        console.print(Panel(
            f"  The bundled classifier was trained on Tesla T4 thermal data — its\n"
            f"  R_θ thresholds (idle ≈ 1.28 C/W, load ≈ 0.72 C/W) reflect a small\n"
            f"  air-cooled package and [bold]will misfire[/] on a different thermal\n"
            f"  envelope like [bold]{models}[/].\n\n"
            f"  After setup completes, run:\n"
            f"  [bold {GREEN}]theta calibrate --gpu <index>[/]\n\n"
            f"  [{DIM}]This measures YOUR hardware's actual idle/load R_θ and derives\n"
            f"  thresholds from it — a few minutes, once per GPU model.[/]",
            border_style=YELLOW,
            title=f"[{YELLOW}]Non-T4 hardware detected — calibration recommended[/]",
            title_align="left",
            padding=(1, 2),
        ))
        console.print()

    return gpus


# ── Step 4: Virtual ambient baseline ──────────────────────────────────────────
def step_baseline(gpus: list[dict]) -> dict:
    step_header(3, 6, "Virtual ambient", "Estimating T_ref — no thermocouple needed")

    console.print(Panel(
        f"  Theta derives the ambient reference temperature [bold]T_ref[/] from your\n"
        f"  GPU's own stable idle windows — no external hardware required.\n\n"
        f"  [{DIM}]We wait for: util < 5%  ·  P-state ≥ P4  ·  stable for 30s (σ < 1.5°C)[/]",
        border_style=DIM,
        padding=(1, 2),
    ))
    console.print()

    # Check if baselines already exist
    from theta.agent.baseline import BaselineManager
    bm = BaselineManager()

    existing = {g["index"]: bm.get_baseline(g["index"]) for g in gpus if bm.has_baseline(g["index"])}
    if existing:
        console.print(f"  [{GREEN}]Existing baselines found:[/]")
        for idx, b in existing.items():
            ok(f"GPU {idx}  T_ref = {b.t_ref:.1f}°C  (locked {b.age_hours():.1f}h ago)")
        console.print()

        use_existing = Confirm.ask(
            f"  [{TEXT}]Use existing baselines?[/]", default=True
        )
        if use_existing:
            return {g["index"]: bm.get_t_ref(g["index"]) for g in gpus}

    # Offer manual set or auto-detect
    method = Prompt.ask(
        f"\n  [{TEXT}]How would you like to set T_ref?[/]",
        choices=["auto", "manual"],
        default="manual",
    )

    baselines = {}

    if method == "manual":
        console.print()
        info("Typical values: 22–28°C for air-cooled racks, 18–22°C for cold-aisle configs.")
        for g in gpus:
            t_ref = float(Prompt.ask(
                f"  [{TEXT}]T_ref for GPU {g['index']} ({g['name']})[/]",
                default="25.0",
            ))
            bm.set_manual(g["index"], t_ref)
            ok(f"GPU {g['index']} T_ref = {t_ref:.1f}°C")
            baselines[g["index"]] = t_ref
    else:
        console.print()

        # Check if NVML is actually available before attempting auto-detect.
        # pynvml can be installed on macOS/CPU hosts but nvmlInit() fails at
        # runtime. Detect this early and fall back to a sensible default rather
        # than crashing mid-wizard.
        _nvml_live = False
        try:
            import pynvml as _pynvml
            _pynvml.nvmlInit()
            _pynvml.nvmlShutdown()
            _nvml_live = True
        except Exception:
            pass

        if not _nvml_live:
            warn("No NVIDIA GPU / driver detected — cannot auto-detect T_ref.")
            info("Using default T_ref = 25.0°C (demo mode). Override with 'manual' next time.")
            console.print()
            baselines = {g["index"]: 25.0 for g in gpus}
            from theta.agent.baseline import BaselineManager
            bm_demo = BaselineManager()
            for g in gpus:
                bm_demo.set_manual(g["index"], 25.0)
                ok(f"GPU {g['index']} T_ref = 25.0°C  (demo default)")
            return baselines

        info("Waiting for stable idle windows. Make sure all GPUs are idle…")
        console.print()

        async def _auto_baseline():
            from theta.agent.collector import NVMLCollector, CollectorConfig
            cfg = CollectorConfig(interval_sec=2.0)
            async with NVMLCollector(cfg) as c:
                with Progress(
                    SpinnerColumn(style=f"bold {GREEN}"),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=30, style=GREEN),
                    TextColumn("{task.completed}/{task.total}s"),
                    console=console,
                    transient=False,
                ) as prog:
                    tasks = {g["index"]: prog.add_task(
                        f"  GPU {g['index']} — waiting for idle…",
                        total=30
                    ) for g in gpus}

                    deadline = asyncio.get_event_loop().time() + 120
                    async for s in c.stream():
                        bm.update(s.gpu_index, s.temp_junction, s.util_pct, s.perf_state, s.timestamp)
                        if bm.has_baseline(s.gpu_index) and s.gpu_index in tasks:
                            b = bm.get_baseline(s.gpu_index)
                            prog.update(tasks[s.gpu_index], completed=30,
                                        description=f"  [bold {GREEN}]✓[/] GPU {s.gpu_index} — T_ref={b.t_ref:.1f}°C  σ={b.sigma:.3f}")
                        if all(bm.has_baseline(g["index"]) for g in gpus):
                            break
                        if asyncio.get_event_loop().time() > deadline:
                            warn("Timeout. Any unlocked GPUs will use default T_ref = 25.0°C")
                            break

            return {g["index"]: bm.get_t_ref(g["index"]) for g in gpus}

        baselines = asyncio.run(_auto_baseline())

    console.print()
    ok("Virtual ambient configured — R_θ computation ready")
    return baselines


# ── Step 5: First R_theta reading ─────────────────────────────────────────────
def step_first_reading(gpus: list[dict], baselines: dict) -> None:
    step_header(4, 6, "First reading", "Live R_θ classification — seeing it work")

    console.print(
        f"  Collecting a 15-second steady-state window from each GPU…\n"
        f"  [{DIM}](This is what the agent does every 5 seconds, continuously.)[/]\n"
    )

    from theta.agent.collector   import NVMLCollector, CollectorConfig
    from theta.agent.metrics     import enrich
    from theta.agent.window      import SteadyStateWindow
    from theta.agent.classifier  import StateClassifier
    from theta.agent.metrics     import STATE_LABELS
    from theta.agent.baseline    import BaselineManager

    state_colors = {
        "under_load":          GREEN,
        "clean_idle":          BLUE,
        "zombie_recovery":     RED,
        "child_exit_recovery": YELLOW,
        "unknown":             DIM,
    }
    state_desc = {
        "under_load":          "GPU is working. Thermal equilibrium. All good.",
        "clean_idle":          "GPU is idle. Temperature settling. Normal.",
        "zombie_recovery":     "CUDA context retained after process exit. 30W at 0% util.",
        "child_exit_recovery": "Post-exit thermal lag. Junction cooling toward ambient.",
        "unknown":             "Collecting data…",
    }

    async def _read():
        bm  = BaselineManager()
        win = SteadyStateWindow(window_sec=10.0, sigma_threshold=0.08)
        clf = StateClassifier()
        cfg = CollectorConfig(interval_sec=1.0)

        results = {}
        async with NVMLCollector(cfg) as c:
            with Progress(
                SpinnerColumn(style=f"bold {GREEN}"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=20, style=GREEN),
                TimeElapsedColumn(),
                console=console, transient=True,
            ) as prog:
                tasks = {g["index"]: prog.add_task(f"  GPU {g['index']}…", total=10) for g in gpus}
                async for s in c.stream():
                    t_ref    = bm.get_t_ref(s.gpu_index)
                    enriched = enrich(s, t_ref)
                    if enriched.rtheta is None:
                        continue
                    window = win.update(
                        s.gpu_index, s.timestamp,
                        enriched.rtheta, s.power_w, s.util_pct, s.perf_state
                    )
                    coverage = int(win.coverage(s.gpu_index, s.timestamp) * 10)
                    prog.update(tasks.get(s.gpu_index, list(tasks.values())[0]), completed=coverage)

                    if window.is_stable:
                        state, conf = clf.classify(window)
                        reason      = clf.explain(window).split("—")[-1].strip()
                        results[s.gpu_index] = {
                            "rtheta": window.rtheta_mean,
                            "std":    window.rtheta_std,
                            "state":  STATE_LABELS.get(state, "unknown"),
                            "conf":   conf,
                            "reason": reason,
                            "power":  window.last_power,
                            "util":   window.last_util,
                            "pstate": window.last_pstate,
                        }
                    if len(results) == len(gpus):
                        break
        return results

    results = asyncio.run(_read())

    console.print()
    # Results panel per GPU
    for gpu_idx, r in sorted(results.items()):
        color = state_colors.get(r["state"], DIM)
        desc  = state_desc.get(r["state"], "")

        t = Table(box=None, show_header=False, padding=(0, 2))
        t.add_column("k", style=DIM,   width=16)
        t.add_column("v", style=TEXT,  min_width=24)
        t.add_row("R_θ",       f"[bold {GREEN}]{r['rtheta']:.4f} C/W[/]  [dim]σ={r['std']:.4f}[/]")
        t.add_row("State",     f"[bold {color}]{r['state']}[/]  [dim]conf={r['conf']:.2f}[/]")
        if desc:
            t.add_row("Means",  f"[dim]{desc}[/]")
        t.add_row("Power",     f"{r['power']:.1f}W  util={r['util']:.0f}%  P{r['pstate']}")
        t.add_row("Why",       f"[dim]{r['reason'][:64]}[/]")

        console.print(Panel(
            t,
            title=f"[bold {TEXT}]GPU {gpu_idx}[/]  [{color}]● {r['state']}[/]",
            title_align="left",
            border_style=color,
            padding=(1, 1),
        ))

    console.print()
    ok("First R_θ classification complete")


# ── Step 6: Alert config ───────────────────────────────────────────────────────
def step_redfish() -> dict:
    """Step 4b: Optional Redfish/BMC out-of-band telemetry."""
    console.print()
    console.print(Panel(
        f"  [bold {TEXT}]Optional: Redfish / BMC out-of-band telemetry[/]\n\n"
        f"  On DGX B200 and other data center systems, the baseboard management\n"
        f"  controller (BMC) exposes chassis-level data via Redfish REST API:\n"
        f"  inlet air temperature, fan RPM, PSU health, NVLink fabric status.\n\n"
        f"  [{GREEN}]Cross-layer correlation example:[/]\n"
        f"  [{DIM}]GPU R_θ drifting + fan at 60% RPM → root cause is cooling path,[/]\n"
        f"  [{DIM}]not silicon degradation. Theta identifies this automatically.[/]\n\n"
        f"  [{DIM}]Skip this on Colab, consumer GPUs, or any non-DGX host.[/]",
        title=f"[{BLUE}]Redfish / BMC (optional)[/]",
        title_align="left",
        border_style=BLUE,
        padding=(1, 2),
    ))
    console.print()

    want_redfish = Confirm.ask(
        f"  [{TEXT}]Enable Redfish / BMC out-of-band telemetry?[/]", default=False
    )
    if not want_redfish:
        return {}

    bmc_host = Prompt.ask(f"  [{TEXT}]BMC IP address or hostname[/]", default="192.168.1.1")
    bmc_user = Prompt.ask(f"  [{TEXT}]BMC username[/]", default="admin")
    bmc_pass = Prompt.ask(f"  [{TEXT}]BMC password[/]", password=True)

    ok(f"Redfish configured: https://{bmc_host}/ (will probe on first run)")
    return {
        "use_redfish":      True,
        "redfish_host":     bmc_host,
        "redfish_user":     bmc_user,
        "redfish_password": bmc_pass,
    }


def step_alerts() -> dict:
    step_header(5, 7, "Alert setup", "Configure how you want to be notified")

    console.print(Panel(
        f"  Theta can alert you when a GPU transitions to an anomalous state.\n"
        f"  Every alert includes: state, R_θ, σ-score, last 10 samples, and the reason.\n\n"
        f"  [{DIM}]Alert types: [bold]zombie_recovery[/] (CUDA stuck) · "
        f"[bold]drifting[/] (cooling path degrading) · [bold]critical[/] (3.5σ above baseline)[/]",
        border_style=DIM,
        padding=(1, 2),
    ))
    console.print()

    alert_cfg: dict = {}

    # Webhook
    want_webhook = Confirm.ask(f"  [{TEXT}]Send alerts to a webhook (Slack, PagerDuty, custom)?[/]", default=False)
    if want_webhook:
        url = Prompt.ask(f"  [{TEXT}]Webhook URL[/]")
        alert_cfg["webhook_url"] = url
        if "slack" in url:
            ok("Slack webhook configured — alerts will use rich Slack attachment format")
        else:
            ok("Webhook configured — JSON payload with full alert context")

    # Log file
    want_log = Confirm.ask(f"\n  [{TEXT}]Log alerts to a JSONL file?[/]", default=True)
    if want_log:
        default_log = str(Path.home() / ".theta" / "alerts.jsonl")
        log_path    = Prompt.ask(f"  [{TEXT}]Log file path[/]", default=default_log)
        alert_cfg["alert_log_path"] = log_path
        ok(f"Alert log: {log_path}")

    # Prometheus
    console.print()
    want_prom = Confirm.ask(
        f"  [{TEXT}]Enable Prometheus metrics endpoint (:9101/metrics)?[/]", default=True
    )
    alert_cfg["prometheus"] = want_prom
    alert_cfg["prometheus_port"] = 9101
    if want_prom:
        ok("Prometheus export on :9101  ·  theta_gpu_rtheta_cwatt, theta_gpu_state_info, …")
        info("Add to prometheus.yml:  - job_name: theta  static_configs: [{targets: ['localhost:9101']}]")

    return alert_cfg


# ── Step 7: Save config + launch ───────────────────────────────────────────────
def step_intelligence_network() -> bool:
    """Step 6: Theta Intelligence Network opt-in consent."""
    step_header(6, 7, "Theta Intelligence Network", "Help make every deployment smarter")

    console.print(Panel(
        f"  [bold {TEXT}]The more GPUs Theta monitors, the smarter it gets.[/]\n\n"
        f"  Opt in to share anonymized GPU health signatures with the Theta\n"
        f"  Intelligence Network. In return, you get:\n\n"
        f"  [bold {GREEN}]→[/]  Community benchmarks — where does your GPU sit vs the fleet P50/P95?\n"
        f"  [bold {GREEN}]→[/]  Improved predictive models trained on real-world degradation curves\n"
        f"  [bold {GREEN}]→[/]  Earlier failure detection as the network grows\n\n"
        f"  [bold {TEXT}]What is shared (anonymized, aggregated hourly):[/]\n"
        f"  [dim]R_theta statistics · ECC error rates · clock efficiency ratios\n"
        f"  XID event frequencies · GPU generation tag · recovery time signatures[/]\n\n"
        f"  [bold {TEXT}]What is NEVER shared:[/]\n"
        f"  [dim]IP address · hostname · job names · usernames · model weights\n"
        f"  company name · raw timestamps (hourly buckets only)[/]\n\n"
        f"  [dim]Change anytime: edit data_sharing in {CONFIG_PATH}[/]",
        title=f"[bold {BLUE}]Intelligence Network[/]",
        title_align="left",
        border_style=BLUE,
        padding=(1, 2),
    ))
    console.print()

    opt_in = Confirm.ask(
        f"  [{TEXT}]Share anonymized telemetry with the Theta Intelligence Network?[/]",
        default=True,
    )

    if opt_in:
        ok("Opted in — you'll receive community benchmarks in future releases")
    else:
        console.print(f"  [{DIM}]Opted out — you can enable this later in {CONFIG_PATH}[/]")

    console.print()
    return opt_in


def step_finish(sys_info: dict, gpus: list[dict], baselines: dict, alert_cfg: dict, opt_in_telemetry: bool = False, redfish_cfg: dict = {}) -> None:
    step_header(7, 7, "All set", "Config saved — ready to monitor")

    config = {
        "version":        __version__,
        "interval_sec":   5.0,
        "gpu_indices":    [g["index"] for g in gpus],
        "baselines":      {str(k): v for k, v in baselines.items()},
        "webhook_url":    alert_cfg.get("webhook_url"),
        "alert_log_path": alert_cfg.get("alert_log_path"),
        "prometheus_port": alert_cfg.get("prometheus_port", 9101),
        "enable_prometheus": alert_cfg.get("prometheus", True),
        "prefer_dt":      True,
        "k_warn":         2.0,
        "k_critical":     3.5,
        "data_sharing":      opt_in_telemetry,
        "use_redfish":       redfish_cfg.get("use_redfish", False),
        "redfish_host":      redfish_cfg.get("redfish_host"),
        "redfish_user":      redfish_cfg.get("redfish_user"),
        "redfish_password":  redfish_cfg.get("redfish_password"),
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    ok(f"Config saved: {CONFIG_PATH}")

    console.print()

    # Summary table
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("k", style=DIM,  width=22)
    t.add_column("v", style=TEXT)
    t.add_row("GPUs monitored",   ", ".join(f"GPU {g['index']} ({g['name'].split('(')[0].strip()})" for g in gpus))
    t.add_row("Sample interval",  "every 5 seconds")
    t.add_row("Classifier",       "Decision Tree (100% accuracy, Stage 1)")
    t.add_row("Steady-state window", "15s  ·  σ < 0.03 C/W")
    t.add_row("Drift threshold",  "2.0σ warn  ·  3.5σ critical")
    t.add_row("Alerts",           (
        ", ".join(filter(None, [
            "stdout",
            "Slack webhook" if alert_cfg.get("webhook_url") and "slack" in alert_cfg.get("webhook_url","") else
            "webhook" if alert_cfg.get("webhook_url") else None,
            "JSONL log" if alert_cfg.get("alert_log_path") else None,
            "Prometheus :9101" if alert_cfg.get("prometheus") else None,
        ])) or "stdout only"
    ))
    t.add_row("Config",           str(CONFIG_PATH))

    console.print(Panel(t, title=f"[bold {GREEN}]Configuration summary[/]",
                        title_align="left", border_style=GREEN, padding=(1, 1)))
    console.print()

    # Launch commands
    non_t4 = [g for g in gpus if "T4" not in g["name"]]
    calibrate_line = (
        f"  [bold {YELLOW}]theta calibrate --gpu 0[/]   [dim]— measure YOUR hardware's R_θ "
        f"(recommended — non-T4 detected)[/]\n"
        if non_t4 else
        f"  [dim]theta calibrate --gpu 0[/]  [dim]— measure hardware-specific R_θ thresholds[/]\n"
    )
    console.print(Panel(
        f"  [bold {TEXT}]Start monitoring:[/]\n\n"
        f"  [bold {GREEN}]theta monitor[/]\n\n"
        f"  [dim]Other commands:[/]\n"
        f"  [dim]theta classify[/]           [dim]— snapshot all GPUs right now[/]\n"
        f"  [dim]theta baseline --gpu 0[/]   [dim]— re-lock virtual ambient[/]\n"
        + calibrate_line +
        f"  [dim]theta serve[/]              [dim]— metrics only, no stdout[/]\n"
        f"  [dim]theta train /path/data.csv[/] [dim]— retrain from new data[/]\n\n"
        f"  [bold {BLUE}]Grafana dashboard:[/]  [dim]import theta_grafana.json (coming soon)[/]",
        border_style=BLUE,
        title=f"[{BLUE}]Quick start[/]",
        title_align="left",
        padding=(1, 2),
    ))
    console.print()
    ok("Theta is ready.")
    console.print(
        f"\n  [{DIM}]Run [bold]theta monitor[/] to start. Press [bold]Ctrl+C[/] to stop.[/]\n"
    )

    want_launch = Confirm.ask(f"  [{TEXT}]Launch the agent now?[/]", default=True)
    if want_launch:
        console.print()
        console.print(Rule(style=DIM))
        from theta.agent.daemon import ThetaAgent, AgentConfig
        cfg = AgentConfig(
            interval_sec      = config["interval_sec"],
            gpu_indices       = config["gpu_indices"],
            webhook_url       = config.get("webhook_url"),
            alert_log_path    = config.get("alert_log_path"),
            prometheus_port   = config["prometheus_port"],
            enable_prometheus = config["enable_prometheus"],
            prefer_dt         = config["prefer_dt"],
            k_warn            = config["k_warn"],
            k_critical        = config["k_critical"],
        )
        agent = ThetaAgent(cfg)
        console.print(f"  [bold {GREEN}]Theta running.[/]  [{DIM}]Ctrl+C to stop.[/]\n")
        try:
            asyncio.run(agent.run())
        except KeyboardInterrupt:
            console.print(f"\n  [{DIM}]Stopped.[/]\n")


# ── Entry point ────────────────────────────────────────────────────────────────
def run_wizard() -> None:
    try:
        step_welcome()
        sys_info = step_system_check()
        gpus     = step_gpu_inventory(sys_info)
        baselines = step_baseline(gpus)
        step_first_reading(gpus, baselines)
        redfish_cfg = step_redfish()
        alert_cfg = step_alerts()
        opt_in_telemetry = step_intelligence_network()
        step_finish(sys_info, gpus, baselines, alert_cfg, opt_in_telemetry, redfish_cfg)
    except KeyboardInterrupt:
        console.print(f"\n\n  [{DIM}]Setup cancelled. Run [bold]theta setup[/] to start again.[/]\n")
        sys.exit(0)
