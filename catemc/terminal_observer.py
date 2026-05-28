"""Terminal-only live dashboard for CatEMC EMC simulations.

Streams the JSONL metrics file and renders an animated Rich + Plotext
dashboard directly in the terminal. Works over SSH — no GUI needed.

Usage
-----
    python -m catemc.terminal_observer path/to/metrics.jsonl [--refresh 2.0] [--history 200]
"""
import json
import time
import argparse
import os
import sys
from collections import defaultdict, deque
from io import StringIO

import numpy as np

try:
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.console import Console
except ImportError:
    sys.exit("ERROR: 'rich' is required.  Install with:  pip install rich")

try:
    import plotext as plt
except ImportError:
    sys.exit("ERROR: 'plotext' is required.  Install with:  pip install plotext")


# ──────────────────────────────────────────────────────────────────────
# JSONL Tail Reader
# ──────────────────────────────────────────────────────────────────────

class JSONLTailer:
    """Poll-based file tailer for JSONL metrics files.

    Reads all existing records on init, then yields new records on each
    call to poll().  Robust to network filesystems (CephFS).
    """

    def __init__(self, path):
        self.path = path
        self._pos = 0
        self._buffer = ""

    def read_all(self):
        """Read all existing generation records from the file."""
        records = []
        if not os.path.exists(self.path):
            return records
        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'gen' in rec:
                    records.append(rec)
            self._pos = f.tell()
        return records

    def poll(self):
        """Read new lines since last read. Returns list of new records."""
        new_records = []
        if not os.path.exists(self.path):
            return new_records
        try:
            with open(self.path, 'r') as f:
                f.seek(self._pos)
                new_data = f.read()
                self._pos = f.tell()
        except (IOError, OSError):
            return new_records

        if not new_data:
            return new_records

        self._buffer += new_data
        lines = self._buffer.split('\n')
        # Keep incomplete last line in buffer
        self._buffer = lines[-1]
        for line in lines[:-1]:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'gen' in rec:
                new_records.append(rec)
        return new_records


# ──────────────────────────────────────────────────────────────────────
# State Accumulator
# ──────────────────────────────────────────────────────────────────────

class DashboardState:
    """Accumulates time-series data from JSONL records."""

    def __init__(self, max_history=200):
        self.max_history = max_history
        self.gens = deque(maxlen=max_history)
        self.elapsed = deque(maxlen=max_history)
        self.acc_rates = deque(maxlen=max_history)
        self.energy_mean = deque(maxlen=max_history)
        self.energy_min = deque(maxlen=max_history)
        self.energy_max = deque(maxlen=max_history)
        self.energy_std = deque(maxlen=max_history)
        self.compositions = defaultdict(lambda: deque(maxlen=max_history))
        self.move_rates = defaultdict(lambda: deque(maxlen=max_history))
        self.move_gens = defaultdict(lambda: deque(maxlen=max_history))
        self.latest_temp_energy = []
        self.latest_record = None
        self.n_records = 0
        self.start_time = None

    def ingest(self, records):
        """Add a batch of records to state."""
        for rec in records:
            self._ingest_one(rec)

    def _ingest_one(self, rec):
        gen = rec['gen']
        self.gens.append(gen)
        self.elapsed.append(rec.get('elapsed_s', 0))
        self.acc_rates.append(rec.get('acc_rate', 0))

        energy = rec.get('energy', {})
        if energy:
            self.energy_mean.append(energy.get('mean', 0))
            self.energy_min.append(energy.get('min', 0))
            self.energy_max.append(energy.get('max', 0))
            self.energy_std.append(energy.get('std', 0))

        comp = rec.get('composition', {})
        for elem, val in comp.items():
            if isinstance(val, dict):
                self.compositions[elem].append(val.get('mean', 0))
            else:
                self.compositions[elem].append(val)

        n_tot = rec.get('n_total', {})
        n_acc = rec.get('n_accepted', {})
        for mt in n_tot:
            t = n_tot[mt]
            if t > 0:
                rate = n_acc.get(mt, 0) / t
                self.move_rates[mt].append(rate)
                self.move_gens[mt].append(gen)

        te = rec.get('temp_energy', [])
        if te:
            self.latest_temp_energy = te

        self.latest_record = rec
        self.n_records += 1
        if self.start_time is None:
            self.start_time = rec.get('t', time.time())


# ──────────────────────────────────────────────────────────────────────
# Plotext Helpers — render plots to string
# ──────────────────────────────────────────────────────────────────────

def _plot_to_string(plot_func, width=60, height=15):
    """Call plot_func() to set up a plotext plot, then return as string."""
    plt.clf()
    plt.clc()
    plt.plotsize(width, height)
    plot_func()
    return plt.build()


def _energy_plot(state, width, height):
    """Energy trajectory: mean with min/max envelope."""
    def _plot():
        if not state.energy_mean:
            plt.title("Energy (no data)")
            return
        gens = list(state.gens)[-len(state.energy_mean):]
        plt.plot(gens, list(state.energy_mean), label="mean", color="blue")
        plt.plot(gens, list(state.energy_min), label="min", color="cyan")
        plt.plot(gens, list(state.energy_max), label="max", color="red")
        plt.title("Energy (eV)")
        plt.xlabel("Generation")
    return _plot_to_string(_plot, width, height)


def _acceptance_plot(state, width, height):
    """Overall acceptance rate over generations."""
    def _plot():
        if not state.acc_rates:
            plt.title("Acceptance (no data)")
            return
        gens = list(state.gens)
        plt.plot(gens, list(state.acc_rates), color="green")
        plt.title("Overall Acceptance Rate")
        plt.xlabel("Generation")
        plt.ylim(0, 1)
    return _plot_to_string(_plot, width, height)


def _move_type_plot(state, width, height):
    """Per-move-type acceptance rates."""
    def _plot():
        if not state.move_rates:
            plt.title("Move Types (no data)")
            return
        colors = ["blue", "red", "green", "magenta", "cyan"]
        for i, (mt, rates) in enumerate(sorted(state.move_rates.items())):
            gens = list(state.move_gens[mt])
            plt.plot(gens, list(rates), label=mt,
                     color=colors[i % len(colors)])
        plt.title("Per-Move Acceptance")
        plt.xlabel("Generation")
        plt.ylim(0, 1)
    return _plot_to_string(_plot, width, height)


def _composition_plot(state, width, height):
    """Composition time-series."""
    def _plot():
        if not state.compositions:
            plt.title("Composition (no data)")
            return
        colors = ["blue", "red", "green", "magenta", "cyan"]
        gens = list(state.gens)
        for i, (elem, fracs) in enumerate(sorted(state.compositions.items())):
            g = gens[-len(fracs):]
            plt.plot(g, list(fracs), label=elem,
                     color=colors[i % len(colors)])
        plt.title("Composition")
        plt.xlabel("Generation")
    return _plot_to_string(_plot, width, height)


def _temp_energy_plot(state, width, height):
    """Temperature vs energy scatter (latest generation)."""
    def _plot():
        te = state.latest_temp_energy
        if not te:
            plt.title("T vs E (no data)")
            return
        temps = [p[0] for p in te]
        energies = [p[1] for p in te]
        plt.scatter(temps, energies, color="magenta", marker="dot")
        plt.title("T vs E (latest gen)")
        plt.xlabel("Temperature (K)")
    return _plot_to_string(_plot, width, height)


def _elapsed_plot(state, width, height):
    """Time per generation."""
    def _plot():
        if not state.elapsed:
            plt.title("Timing (no data)")
            return
        gens = list(state.gens)
        plt.plot(gens, list(state.elapsed), color="blue")
        if state.elapsed:
            mean_t = np.mean(list(state.elapsed))
            plt.hline(mean_t, color="red")
        plt.title(f"Time/Gen (mean={np.mean(list(state.elapsed)):.1f}s)")
        plt.xlabel("Generation")
    return _plot_to_string(_plot, width, height)


# ──────────────────────────────────────────────────────────────────────
# Dashboard Layout
# ──────────────────────────────────────────────────────────────────────

def _build_summary_table(state):
    """Build the summary stats table."""
    table = Table(title="Summary", expand=True, show_lines=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    rec = state.latest_record
    if rec is None:
        table.add_row("Status", "Waiting for data...")
        return table

    gen = rec.get('gen', 0)
    elapsed = rec.get('elapsed_s', 0)
    acc = rec.get('acc_rate', 0)
    energy = rec.get('energy', {})
    t = rec.get('t', 0)

    table.add_row("Generation", f"{gen}")
    table.add_row("Records", f"{state.n_records}")
    table.add_row("Last Step", f"{elapsed:.1f}s")
    table.add_row("Acceptance", f"{acc:.1%}")
    if energy:
        table.add_row("E mean", f"{energy['mean']:.2f} eV")
        table.add_row("E min", f"{energy['min']:.2f} eV")
        table.add_row("E max", f"{energy['max']:.2f} eV")
        table.add_row("E std", f"{energy['std']:.3f} eV")
    if t:
        table.add_row("Last Update", time.strftime('%H:%M:%S', time.localtime(t)))

    # Total runtime
    if state.start_time:
        runtime = t - state.start_time
        h, rem = divmod(int(runtime), 3600)
        m, s = divmod(rem, 60)
        table.add_row("Runtime", f"{h}h {m}m {s}s")

    return table


def _build_moves_table(state):
    """Build the per-move-type acceptance table."""
    table = Table(title="Move Types", expand=True, show_lines=False)
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Accepted", style="green", justify="right")
    table.add_column("Total", style="white", justify="right")
    table.add_column("Rate", style="yellow", justify="right")

    rec = state.latest_record
    if rec is None:
        return table

    n_acc = rec.get('n_accepted', {})
    n_tot = rec.get('n_total', {})

    for mt in sorted(set(list(n_tot.keys()) + list(n_acc.keys()))):
        a = n_acc.get(mt, 0)
        t = n_tot.get(mt, 0)
        rate = a / t if t > 0 else 0
        table.add_row(mt, str(a), str(t), f"{rate:.1%}")

    return table


def _build_composition_table(state):
    """Build composition summary table."""
    table = Table(title="Composition", expand=True, show_lines=False)
    table.add_column("Species", style="cyan", no_wrap=True)
    table.add_column("Mean Frac", style="white", justify="right")
    table.add_column("Std", style="dim", justify="right")

    rec = state.latest_record
    if rec is None:
        return table

    comp = rec.get('composition', {})
    for elem in sorted(comp.keys()):
        val = comp[elem]
        if isinstance(val, dict):
            table.add_row(elem, f"{val.get('mean', 0):.4f}",
                          f"±{val.get('std', 0):.4f}")
        else:
            table.add_row(elem, f"{val:.4f}", "—")

    return table


def build_dashboard(state, console_width=None, console_height=None):
    """Build the full Rich Layout dashboard."""
    # Adaptive sizing
    w = console_width or 120
    h = console_height or 40

    # Plot dimensions (in characters)
    plot_w = max(30, (w - 6) // 2)
    plot_h = max(8, (h - 14) // 3)

    layout = Layout()

    # Top row: summary tables
    layout.split_column(
        Layout(name="top", size=12),
        Layout(name="mid", size=plot_h + 2),
        Layout(name="bot", size=plot_h + 2),
    )

    # Top row: 3 tables side by side
    layout["top"].split_row(
        Layout(Panel(_build_summary_table(state), border_style="blue"), name="summary"),
        Layout(Panel(_build_moves_table(state), border_style="green"), name="moves"),
        Layout(Panel(_build_composition_table(state), border_style="magenta"), name="comp"),
    )

    # Middle row: energy + acceptance plots
    energy_str = _energy_plot(state, plot_w, plot_h)
    acc_str = _acceptance_plot(state, plot_w, plot_h)
    layout["mid"].split_row(
        Layout(Panel(Text(energy_str), title="Energy Trajectory",
                     border_style="blue"), name="energy_plot"),
        Layout(Panel(Text(acc_str), title="Acceptance Rate",
                     border_style="green"), name="acc_plot"),
    )

    # Bottom row: move types + T vs E
    moves_str = _move_type_plot(state, plot_w, plot_h)
    te_str = _temp_energy_plot(state, plot_w, plot_h)
    layout["bot"].split_row(
        Layout(Panel(Text(moves_str), title="Per-Move Acceptance",
                     border_style="yellow"), name="moves_plot"),
        Layout(Panel(Text(te_str), title="T vs E (latest)",
                     border_style="magenta"), name="te_plot"),
    )

    return layout


# ──────────────────────────────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────────────────────────────

def run(metrics_file, refresh=2.0, history=200):
    """Run the live terminal dashboard."""
    console = Console()

    # Wait for file
    if not os.path.exists(metrics_file):
        console.print(f"[yellow]Waiting for metrics file:[/] {metrics_file}")
        while not os.path.exists(metrics_file):
            time.sleep(1)

    tailer = JSONLTailer(metrics_file)
    state = DashboardState(max_history=history)

    # Initial read of all existing data
    console.print(f"[cyan]Reading existing records from:[/] {metrics_file}")
    existing = tailer.read_all()
    state.ingest(existing)
    console.print(f"[green]Loaded {len(existing)} records. Starting live dashboard...[/]")
    time.sleep(0.5)

    try:
        with Live(build_dashboard(state, console.width, console.height),
                  console=console, refresh_per_second=1/refresh,
                  screen=True) as live:
            while True:
                time.sleep(refresh)
                new_records = tailer.poll()
                if new_records:
                    state.ingest(new_records)
                live.update(
                    build_dashboard(state, console.width, console.height)
                )
    except KeyboardInterrupt:
        pass
    finally:
        console.print(f"\n[cyan]Dashboard stopped.[/] {state.n_records} total records observed.")


def main():
    parser = argparse.ArgumentParser(
        description='Terminal-only live dashboard for CatEMC metrics (Rich + Plotext)')
    parser.add_argument('metrics_file', help='Path to the JSONL metrics file')
    parser.add_argument('--refresh', type=float, default=2.0,
                        help='Refresh interval in seconds (default: 2.0)')
    parser.add_argument('--history', type=int, default=200,
                        help='Number of generations to keep in rolling history (default: 200)')
    args = parser.parse_args()
    run(args.metrics_file, refresh=args.refresh, history=args.history)


if __name__ == '__main__':
    main()
