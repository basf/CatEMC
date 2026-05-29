"""Multi-system overview dashboard for CatEMC.

Auto-discovers *_metrics.jsonl files in subdirectories and renders a
full-screen live Rich panel showing progress, ETA, and key stats for
all concurrent EMC runs.

Usage:
    catemc overview [--dir PATH] [--refresh 5.0] [--depth 2] [--once]
    python -m catemc.multi_overview [--dir PATH] [--refresh 5.0]
"""
import os
import sys
import time
import argparse
import glob

try:
    import yaml
except ImportError:
    yaml = None

try:
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.console import Console
    from rich.layout import Layout
except ImportError:
    sys.exit("ERROR: 'rich' is required.  Install with:  pip install rich")

from catemc.terminal_observer import JSONLTailer, DashboardState


# ──────────────────────────────────────────────────────────────────────
# Status classification
# ──────────────────────────────────────────────────────────────────────

STATUS_RUNNING = "RUNNING"
STATUS_STALLED = "STALLED"
STATUS_STOPPED = "STOPPED"
STATUS_EMPTY = "WAITING"

STATUS_STYLE = {
    STATUS_RUNNING: "bold green",
    STATUS_STALLED: "bold yellow",
    STATUS_STOPPED: "bold red",
    STATUS_EMPTY: "dim",
}

STALE_THRESHOLD = 300   # seconds (5 min) → STALLED
DEAD_THRESHOLD = 3600   # seconds (1 hr)  → STOPPED


def _classify_status(state):
    """Classify run status from last record timestamp."""
    if state.latest_record is None:
        return STATUS_EMPTY
    last_t = state.latest_record.get('t', 0)
    age = time.time() - last_t
    if age < STALE_THRESHOLD:
        return STATUS_RUNNING
    elif age < DEAD_THRESHOLD:
        return STATUS_STALLED
    else:
        return STATUS_STOPPED


# ──────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────

def discover_jsonl_files(base_dir, max_depth=3):
    """Find all *_metrics.jsonl files within max_depth subdirectories.

    Returns list of (system_name, jsonl_path, config_path_or_None).
    """
    results = []
    base_dir = os.path.abspath(base_dir)

    for root, dirs, files in os.walk(base_dir):
        # Enforce depth limit
        depth = root[len(base_dir):].count(os.sep)
        if depth > max_depth:
            dirs.clear()
            continue

        for fname in files:
            if fname.endswith('_metrics.jsonl'):
                jsonl_path = os.path.join(root, fname)
                system_name = fname.replace('_metrics.jsonl', '')

                # Look for config YAML in same directory or parent
                config_path = None
                for cfg_name in ['emc_config.yaml', 'config.yaml']:
                    candidate = os.path.join(root, cfg_name)
                    if os.path.exists(candidate):
                        config_path = candidate
                        break
                if config_path is None:
                    # Try parent directory
                    parent = os.path.dirname(root)
                    for cfg_name in ['emc_config.yaml', 'config.yaml']:
                        candidate = os.path.join(parent, cfg_name)
                        if os.path.exists(candidate):
                            config_path = candidate
                            break

                results.append((system_name, jsonl_path, config_path))

    return sorted(results, key=lambda x: x[0])


def _read_total_steps(config_path):
    """Read total_steps from an emc_config.yaml. Returns int or None."""
    if config_path is None or yaml is None:
        return None
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get('emc', {}).get('total_steps', None)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# Progress bar (unicode)
# ──────────────────────────────────────────────────────────────────────

_BAR_WIDTH = 20
_BAR_FILL = "━"
_BAR_EMPTY = "╺"


def _progress_bar(current, total):
    """Render a unicode progress bar string."""
    if total is None or total <= 0:
        return ""
    frac = min(current / total, 1.0)
    filled = int(frac * _BAR_WIDTH)
    empty = _BAR_WIDTH - filled
    pct = frac * 100
    bar = f"[green]{_BAR_FILL * filled}[/green][dim]{_BAR_EMPTY * empty}[/dim]"
    return f"{bar} {pct:5.1f}%"


def _format_eta(current_gen, total_steps, avg_elapsed_s):
    """Compute ETA string from progress and average time per generation."""
    if total_steps is None or total_steps <= 0 or avg_elapsed_s <= 0:
        return "—"
    remaining = max(total_steps - current_gen, 0)
    eta_s = remaining * avg_elapsed_s
    if eta_s < 60:
        return f"{eta_s:.0f}s"
    elif eta_s < 3600:
        return f"{eta_s / 60:.0f}m"
    elif eta_s < 86400:
        h = int(eta_s // 3600)
        m = int((eta_s % 3600) // 60)
        return f"{h}h {m}m"
    else:
        d = int(eta_s // 86400)
        h = int((eta_s % 86400) // 3600)
        return f"{d}d {h}h"


def _format_runtime(state):
    """Format total runtime since first record."""
    if state.start_time is None:
        return "—"
    runtime = time.time() - state.start_time
    if runtime < 60:
        return f"{runtime:.0f}s"
    elif runtime < 3600:
        return f"{runtime / 60:.0f}m"
    else:
        h = int(runtime // 3600)
        m = int((runtime % 3600) // 60)
        return f"{h}h {m}m"


# ──────────────────────────────────────────────────────────────────────
# Table builder
# ──────────────────────────────────────────────────────────────────────

def _sort_key(item, sort_by):
    """Return sort key for a (name, state, total_steps) tuple."""
    name, state, total_steps = item
    status = _classify_status(state)
    status_order = {STATUS_RUNNING: 0, STATUS_STALLED: 1, STATUS_STOPPED: 2, STATUS_EMPTY: 3}

    if sort_by == 'status':
        return (status_order.get(status, 9), name)
    elif sort_by == 'gen':
        gen = state.gens[-1] if state.gens else -1
        return (-gen, name)
    elif sort_by == 'energy':
        e = state.energy_mean[-1] if state.energy_mean else float('inf')
        return (e, name)
    elif sort_by == 'acceptance':
        acc = state.acc_rates[-1] if state.acc_rates else -1
        return (-acc, name)
    else:  # 'name'
        return (name,)


def build_overview_table(systems, sort_by='status'):
    """Build the multi-system overview Rich Table.

    Parameters
    ----------
    systems : list of (name, DashboardState, total_steps_or_None)
    sort_by : str
        One of 'status', 'name', 'gen', 'energy', 'acceptance'
    """
    table = Table(
        title="[bold cyan]CatEMC Multi-System Overview[/bold cyan]",
        expand=True,
        show_lines=True,
        header_style="bold",
    )
    table.add_column("System", style="bold white", no_wrap=True, min_width=18)
    table.add_column("Status", justify="center", min_width=8)
    table.add_column("Gen", justify="right", min_width=10)
    table.add_column("Progress", min_width=28)
    table.add_column("ETA", justify="right", min_width=7)
    table.add_column("Acc%", justify="right", min_width=6)
    table.add_column("Ē (eV)", justify="right", min_width=10)
    table.add_column("σ(E)", justify="right", min_width=7)
    table.add_column("t/gen", justify="right", min_width=6)
    table.add_column("Runtime", justify="right", min_width=8)

    sorted_systems = sorted(systems, key=lambda x: _sort_key(x, sort_by))

    n_running = 0
    n_stalled = 0
    n_stopped = 0

    for name, state, total_steps in sorted_systems:
        status = _classify_status(state)
        style = STATUS_STYLE[status]

        if status == STATUS_RUNNING:
            n_running += 1
        elif status == STATUS_STALLED:
            n_stalled += 1
        elif status == STATUS_STOPPED:
            n_stopped += 1

        # Generation
        gen = state.gens[-1] if state.gens else 0
        if total_steps:
            gen_str = f"{gen}/{total_steps}"
        else:
            gen_str = f"{gen}/?"

        # Progress bar
        progress = _progress_bar(gen, total_steps)

        # ETA
        avg_elapsed = 0
        if state.elapsed:
            avg_elapsed = sum(state.elapsed) / len(state.elapsed)
        eta = _format_eta(gen, total_steps, avg_elapsed)

        # Acceptance
        acc = f"{state.acc_rates[-1]:.1%}" if state.acc_rates else "—"

        # Energy
        e_mean = f"{state.energy_mean[-1]:.1f}" if state.energy_mean else "—"
        e_std = f"{state.energy_std[-1]:.2f}" if state.energy_std else "—"

        # Time per gen (latest)
        t_gen = f"{state.elapsed[-1]:.1f}s" if state.elapsed else "—"

        # Runtime
        runtime = _format_runtime(state)

        table.add_row(
            name,
            f"[{style}]{status}[/{style}]",
            gen_str,
            progress,
            eta,
            acc,
            e_mean,
            e_std,
            t_gen,
            runtime,
        )

    # Summary footer
    total = len(systems)
    table.caption = (
        f"[bold]{total}[/bold] systems │ "
        f"[green]{n_running} running[/green] │ "
        f"[yellow]{n_stalled} stalled[/yellow] │ "
        f"[red]{n_stopped} stopped[/red]"
    )

    return table


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def run(base_dir='.', refresh=5.0, depth=3, sort_by='status', once=False,
        stale_timeout=300):
    """Run the multi-system overview dashboard."""
    global STALE_THRESHOLD
    STALE_THRESHOLD = stale_timeout

    console = Console()

    # Discover systems
    console.print(f"[cyan]Scanning for metrics in:[/cyan] {os.path.abspath(base_dir)}")
    discovered = discover_jsonl_files(base_dir, max_depth=depth)

    if not discovered:
        console.print("[red]No *_metrics.jsonl files found.[/red]")
        console.print(f"  Searched {os.path.abspath(base_dir)} (depth={depth})")
        sys.exit(1)

    console.print(f"[green]Found {len(discovered)} systems[/green]")

    # Initialize tailers and states
    tailers = {}
    states = {}
    total_steps_map = {}

    for name, jsonl_path, config_path in discovered:
        tailer = JSONLTailer(jsonl_path)
        records = tailer.read_all()
        state = DashboardState(max_history=50)  # smaller history for overview
        state.ingest(records)
        tailers[name] = tailer
        states[name] = state
        total_steps_map[name] = _read_total_steps(config_path)

    # Build system list
    def _get_systems():
        return [(name, states[name], total_steps_map[name]) for name in states]

    if once:
        table = build_overview_table(_get_systems(), sort_by=sort_by)
        console.print(table)
        return

    # Live full-screen dashboard
    console.print(f"[dim]Refresh: {refresh}s │ Ctrl+C to exit[/dim]\n")
    time.sleep(0.5)

    try:
        with Live(
            build_overview_table(_get_systems(), sort_by=sort_by),
            console=console,
            refresh_per_second=1 / refresh,
            screen=True,
        ) as live:
            while True:
                time.sleep(refresh)
                # Poll all tailers
                for name, tailer in tailers.items():
                    new = tailer.poll()
                    if new:
                        states[name].ingest(new)
                live.update(
                    build_overview_table(_get_systems(), sort_by=sort_by)
                )
    except KeyboardInterrupt:
        pass

    console.print(f"\n[cyan]Overview stopped.[/cyan] {len(states)} systems monitored.")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-system overview dashboard for CatEMC runs')
    parser.add_argument('--dir', '-d', default='.',
                        help='Base directory to scan (default: cwd)')
    parser.add_argument('--refresh', '-r', type=float, default=5.0,
                        help='Refresh interval in seconds (default: 5.0)')
    parser.add_argument('--depth', type=int, default=3,
                        help='Max subdirectory depth to search (default: 2)')
    parser.add_argument('--sort', '-s', default='status',
                        choices=['status', 'name', 'gen', 'energy', 'acceptance'],
                        help='Sort order (default: status)')
    parser.add_argument('--once', action='store_true',
                        help='Print once and exit (no live updates)')
    parser.add_argument('--stale-timeout', type=int, default=300,
                        help='Seconds before a run is marked STALLED (default: 300)')
    args = parser.parse_args()
    run(base_dir=args.dir, refresh=args.refresh, depth=args.depth,
        sort_by=args.sort, once=args.once, stale_timeout=args.stale_timeout)


if __name__ == '__main__':
    main()
