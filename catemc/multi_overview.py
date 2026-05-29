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


def _progress_bar(current, total, is_running=False):
    """Render a unicode progress bar string."""
    if total is None or total <= 0:
        return ""
    frac = min(current / total, 1.0)
    filled = int(frac * _BAR_WIDTH)
    empty = _BAR_WIDTH - filled
    pct = frac * 100
    if is_running and filled < _BAR_WIDTH:
        bar = f"[green]{_BAR_FILL * filled}[/green][blink bold bright_white]╸[/blink bold bright_white][dim]{_BAR_EMPTY * max(0, empty - 1)}[/dim]"
    else:
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


def build_table_view(systems, sort_by='status', console=None):
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
        progress = _progress_bar(gen, total_steps, is_running=(status == STATUS_RUNNING))

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
# Dashboard View (cards + pie chart)
# ──────────────────────────────────────────────────────────────────────

try:
    import plotext as plt
    _HAS_PLOTEXT = True
except ImportError:
    _HAS_PLOTEXT = False


def _render_pie_chart(systems, width=40, height=12):
    """Render a status pie chart as string using plotext."""
    if not _HAS_PLOTEXT:
        return "  (install plotext for pie chart)"

    counts = {STATUS_RUNNING: 0, STATUS_STALLED: 0, STATUS_STOPPED: 0, STATUS_EMPTY: 0}
    for name, state, ts in systems:
        status = _classify_status(state)
        counts[status] += 1

    # Filter out zero counts
    labels = []
    values = []
    colors = []
    color_map = {STATUS_RUNNING: "green", STATUS_STALLED: "yellow",
                 STATUS_STOPPED: "red", STATUS_EMPTY: "gray"}
    for st in [STATUS_RUNNING, STATUS_STALLED, STATUS_STOPPED, STATUS_EMPTY]:
        if counts[st] > 0:
            labels.append(f"{st} ({counts[st]})")
            values.append(counts[st])
            colors.append(color_map[st])

    plt.clf()
    plt.clc()
    plt.plotsize(width, height)
    plt.theme("clear")
    plt.simple_bar(labels, values, color=colors, width=width - 5)
    plt.title("Status Distribution")
    return plt.build()


def _render_system_card(name, state, total_steps, card_width=38):
    """Render a single system card as a Rich Panel."""
    status = _classify_status(state)
    style = STATUS_STYLE[status]

    gen = state.gens[-1] if state.gens else 0
    acc = state.acc_rates[-1] if state.acc_rates else 0
    e_mean = state.energy_mean[-1] if state.energy_mean else None
    e_std = state.energy_std[-1] if state.energy_std else None
    t_gen = state.elapsed[-1] if state.elapsed else 0
    avg_elapsed = sum(state.elapsed) / len(state.elapsed) if state.elapsed else 0

    # Progress bar
    if total_steps and total_steps > 0:
        frac = min(gen / total_steps, 1.0)
        bar_w = card_width - 12
        filled = int(frac * bar_w)
        empty = bar_w - filled
        pct_str = f"{frac*100:.0f}%"
        if status == STATUS_RUNNING and filled < bar_w:
            bar_line = f"[green]{'█' * filled}[/green][blink bold bright_white on green]▌[/blink bold bright_white on green][dim]{'░' * max(0, empty - 1)}[/dim] {pct_str}"
        else:
            bar_line = f"[green]{'█' * filled}[/green][dim]{'░' * empty}[/dim] {pct_str}"
        gen_line = f"Gen [bold]{gen}[/bold] / {total_steps}"
    else:
        bar_line = ""
        gen_line = f"Gen [bold]{gen}[/bold] / ?"

    # ETA
    eta = _format_eta(gen, total_steps, avg_elapsed)
    runtime = _format_runtime(state)

    # Energy
    if e_mean is not None:
        e_line = f"Ē = [cyan]{e_mean:.1f}[/cyan] ± {e_std:.2f} eV"
    else:
        e_line = "Ē = —"

    # Build card content
    lines = [
        f"[{style}]● {status}[/{style}]  ETA: [bold]{eta}[/bold]",
        "",
        gen_line,
        bar_line,
        "",
        f"Acc: [yellow]{acc:.1%}[/yellow]  t/gen: {t_gen:.1f}s",
        e_line,
        f"Runtime: {runtime}",
    ]

    card_text = "\n".join(lines)

    border_color = {"RUNNING": "green", "STALLED": "yellow",
                    "STOPPED": "red", "WAITING": "dim"}.get(status, "white")

    return Panel(
        Text.from_markup(card_text),
        title=f"[bold]{name}[/bold]",
        border_style=border_color,
        width=card_width,
        padding=(0, 1),
    )


def _compute_global_stats(systems):
    """Compute aggregate stats across all systems."""
    n_running = sum(1 for _, s, _ in systems if _classify_status(s) == STATUS_RUNNING)
    n_stalled = sum(1 for _, s, _ in systems if _classify_status(s) == STATUS_STALLED)
    n_stopped = sum(1 for _, s, _ in systems if _classify_status(s) == STATUS_STOPPED)
    total = len(systems)

    # Average progress
    progs = []
    for _, state, ts in systems:
        if ts and ts > 0 and state.gens:
            progs.append(state.gens[-1] / ts)
    avg_progress = sum(progs) / len(progs) if progs else 0

    # Total runtime (max across all)
    max_runtime = 0
    for _, state, _ in systems:
        if state.start_time:
            rt = time.time() - state.start_time
            max_runtime = max(max_runtime, rt)

    return {
        'total': total,
        'running': n_running,
        'stalled': n_stalled,
        'stopped': n_stopped,
        'avg_progress': avg_progress,
        'max_runtime': max_runtime,
    }


def _build_header(stats):
    """Build the summary header panel."""
    h = int(stats['max_runtime'] // 3600)
    m = int((stats['max_runtime'] % 3600) // 60)

    text = (
        f"[bold cyan]CatEMC Multi-System Dashboard[/bold cyan]\n\n"
        f"  Systems: [bold]{stats['total']}[/bold]  │  "
        f"[green]● {stats['running']} running[/green]  "
        f"[yellow]● {stats['stalled']} stalled[/yellow]  "
        f"[red]● {stats['stopped']} stopped[/red]\n"
        f"  Avg Progress: [bold]{stats['avg_progress']:.0%}[/bold]  │  "
        f"Max Runtime: [bold]{h}h {m}m[/bold]"
    )
    return Panel(Text.from_markup(text), border_style="cyan", padding=(0, 1))


def build_dashboard_view(systems, sort_by='status', console=None):
    """Build the fancy cards + chart dashboard layout."""
    from rich.columns import Columns
    from rich.console import Group

    sorted_systems = sorted(systems, key=lambda x: _sort_key(x, sort_by))
    stats = _compute_global_stats(systems)

    # Header
    header = _build_header(stats)

    # Determine card width based on terminal
    term_w = console.width if console else 120
    card_width = max(36, min(42, (term_w - 4) // 3))
    ncols = max(1, (term_w - 2) // (card_width + 1))

    # Build cards
    cards = []
    for name, state, total_steps in sorted_systems:
        cards.append(_render_system_card(name, state, total_steps, card_width))

    # Pie chart / status bar
    if _HAS_PLOTEXT and len(systems) > 1:
        pie_str = _render_pie_chart(systems, width=term_w - 4, height=6)
        pie_panel = Panel(
            Text.from_ansi(pie_str, no_wrap=True),
            title="[bold]Status Overview[/bold]",
            border_style="blue",
        )
    else:
        pie_panel = Text("")

    # Grid of cards
    card_grid = Columns(cards, equal=True, expand=True)

    # Compose
    return Group(header, pie_panel, card_grid)


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def run(base_dir='.', refresh=5.0, depth=3, sort_by='status', once=False,
        stale_timeout=300, tabular=False):
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

    _build = build_table_view if tabular else build_dashboard_view

    if once:
        console.print(_build(_get_systems(), sort_by=sort_by, console=console))
        return

    # Live full-screen dashboard
    console.print(f"[dim]Refresh: {refresh}s │ Ctrl+C to exit[/dim]\n")
    time.sleep(0.5)

    try:
        with Live(
            _build(_get_systems(), sort_by=sort_by, console=console),
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
                    _build(_get_systems(), sort_by=sort_by, console=console)
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
        sort_by=args.sort, once=args.once, stale_timeout=args.stale_timeout,
        tabular=args.tabular)


if __name__ == '__main__':
    main()
