"""Live observer dashboard for CatEMC EMC simulations.

Tails the JSONL metrics file produced by EMCLogger and renders a
6-panel matplotlib figure that auto-updates.

Usage
-----
    python -m catemc.observer path/to/metrics.jsonl [--interval 2.0]
"""
import json
import time
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def _parse_records(path):
    """Read all generation records from a JSONL metrics file."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only generation records have 'gen' key
            if 'gen' in rec:
                records.append(rec)
    return records


def _setup_figure():
    """Create the 6-panel figure layout."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('CatEMC Live Observer', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0, w_pad=2.0)

    ax_time, ax_acc = axes[0]
    ax_energy, ax_comp = axes[1]
    ax_moves, ax_temp_e = axes[2]

    ax_time.set_xlabel('Generation')
    ax_time.set_ylabel('Time (s)')
    ax_time.set_title('Time per Generation')

    ax_acc.set_xlabel('Generation')
    ax_acc.set_ylabel('Acceptance Rate')
    ax_acc.set_title('Overall Acceptance Rate')
    ax_acc.set_ylim(-0.05, 1.05)

    ax_energy.set_xlabel('Generation')
    ax_energy.set_ylabel('Energy (eV)')
    ax_energy.set_title('Energy Trajectory')

    ax_comp.set_xlabel('Generation')
    ax_comp.set_ylabel('Fraction')
    ax_comp.set_title('Composition Drift')

    ax_moves.set_xlabel('Generation')
    ax_moves.set_ylabel('Acceptance Rate')
    ax_moves.set_title('Per-Move-Type Acceptance')
    ax_moves.set_ylim(-0.05, 1.05)

    ax_temp_e.set_xlabel('Temperature (K)')
    ax_temp_e.set_ylabel('Energy (eV)')
    ax_temp_e.set_title('Temperature vs Energy (latest gen)')

    return fig, (ax_time, ax_acc, ax_energy, ax_comp, ax_moves, ax_temp_e)


def _update(frame, path, axes, state):
    """Animation update function — re-reads file and redraws."""
    records = _parse_records(path)
    if not records:
        return

    # Only redraw if new data arrived
    if len(records) == state.get('n_records', 0):
        return
    state['n_records'] = len(records)

    ax_time, ax_acc, ax_energy, ax_comp, ax_moves, ax_temp_e = axes

    gens = [r['gen'] for r in records]
    elapsed = [r.get('elapsed_s', 0) for r in records]
    acc_rates = [r.get('acc_rate', 0) for r in records]

    # --- Panel 1: Time per generation ---
    ax_time.clear()
    ax_time.plot(gens, elapsed, 'b-', linewidth=0.8)
    ax_time.set_xlabel('Generation')
    ax_time.set_ylabel('Time (s)')
    ax_time.set_title('Time per Generation')
    if elapsed:
        ax_time.axhline(np.mean(elapsed), color='r', linestyle='--',
                        linewidth=0.7, label=f'mean={np.mean(elapsed):.1f}s')
        ax_time.legend(fontsize=8)

    # --- Panel 2: Overall acceptance rate ---
    ax_acc.clear()
    ax_acc.plot(gens, acc_rates, 'g-', linewidth=0.8)
    ax_acc.set_xlabel('Generation')
    ax_acc.set_ylabel('Acceptance Rate')
    ax_acc.set_title('Overall Acceptance Rate')
    ax_acc.set_ylim(-0.05, 1.05)

    # --- Panel 3: Energy trajectory ---
    ax_energy.clear()
    e_mean = [r['energy']['mean'] for r in records if 'energy' in r]
    e_min = [r['energy']['min'] for r in records if 'energy' in r]
    e_max = [r['energy']['max'] for r in records if 'energy' in r]
    e_gens = [r['gen'] for r in records if 'energy' in r]
    if e_mean:
        ax_energy.plot(e_gens, e_mean, 'b-', linewidth=0.8, label='mean')
        ax_energy.fill_between(e_gens, e_min, e_max, alpha=0.15, color='blue')
        ax_energy.legend(fontsize=8)
    ax_energy.set_xlabel('Generation')
    ax_energy.set_ylabel('Energy (eV)')
    ax_energy.set_title('Energy Trajectory')

    # --- Panel 4: Composition drift ---
    ax_comp.clear()
    comp_data = defaultdict(list)
    comp_gens = []
    for r in records:
        if 'composition' in r and r['composition']:
            comp_gens.append(r['gen'])
            for elem, val in r['composition'].items():
                if isinstance(val, dict):
                    comp_data[elem].append(val.get('mean', 0))
                else:
                    comp_data[elem].append(val)
    if comp_data:
        for elem, fracs in sorted(comp_data.items()):
            ax_comp.plot(comp_gens[:len(fracs)], fracs, linewidth=0.8, label=elem)
        ax_comp.legend(fontsize=7, ncol=2)
    ax_comp.set_xlabel('Generation')
    ax_comp.set_ylabel('Fraction')
    ax_comp.set_title('Composition Drift')

    # --- Panel 5: Per-move-type acceptance ---
    ax_moves.clear()
    move_types = set()
    for r in records:
        if 'n_total' in r:
            move_types.update(r['n_total'].keys())
    move_rates = {mt: [] for mt in move_types}
    move_gens_map = {mt: [] for mt in move_types}
    for r in records:
        n_tot = r.get('n_total', {})
        n_acc = r.get('n_accepted', {})
        for mt in move_types:
            t = n_tot.get(mt, 0)
            if t > 0:
                move_rates[mt].append(n_acc.get(mt, 0) / t)
                move_gens_map[mt].append(r['gen'])
    for mt in sorted(move_types):
        if move_rates[mt]:
            ax_moves.plot(move_gens_map[mt], move_rates[mt],
                          linewidth=0.8, label=mt)
    if move_types:
        ax_moves.legend(fontsize=7)
    ax_moves.set_xlabel('Generation')
    ax_moves.set_ylabel('Acceptance Rate')
    ax_moves.set_title('Per-Move-Type Acceptance')
    ax_moves.set_ylim(-0.05, 1.05)

    # --- Panel 6: Temperature vs Energy (latest generation) ---
    ax_temp_e.clear()
    latest = records[-1]
    te = latest.get('temp_energy', [])
    if te:
        temps = [p[0] for p in te]
        energies = [p[1] for p in te]
        ax_temp_e.scatter(temps, energies, s=12, alpha=0.7, c='purple')
    ax_temp_e.set_xlabel('Temperature (K)')
    ax_temp_e.set_ylabel('Energy (eV)')
    ax_temp_e.set_title('Temperature vs Energy (latest gen)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])


def _print_status(path):
    """Print a text-mode status summary (for headless/SSH use)."""
    records = _parse_records(path)
    if not records:
        print("No generation records yet.")
        return
    latest = records[-1]
    n = len(records)
    gen = latest['gen']
    elapsed = latest.get('elapsed_s', 0)
    acc = latest.get('acc_rate', 0)
    energy = latest.get('energy', {})
    e_str = f"E={energy.get('mean', 0):.3f}" if energy else "E=N/A"
    moves = latest.get('n_total', {})
    n_acc = latest.get('n_accepted', {})
    move_parts = []
    for mt in sorted(moves.keys()):
        t = moves[mt]
        a = n_acc.get(mt, 0)
        move_parts.append(f"{mt}:{a}/{t}")
    comp = latest.get('composition', {})
    if comp:
        parts = []
        for k, v in sorted(comp.items()):
            if isinstance(v, dict):
                parts.append(f"{k}={v.get('mean', 0):.3f}")
            else:
                parts.append(f"{k}={v:.3f}")
        comp_str = " ".join(parts)
    else:
        comp_str = ""

    print(f"Gen {gen:>5d} | {elapsed:>6.1f}s | acc {acc*100:>4.1f}% | {e_str} | {' '.join(move_parts)}")
    if comp_str:
        print(f"         comp: {comp_str}")
    print(f"         ({n} records total, latest at {time.strftime('%H:%M:%S', time.localtime(latest.get('t', 0)))})")


def main():
    parser = argparse.ArgumentParser(
        description='Live observer dashboard for CatEMC metrics.')
    parser.add_argument('metrics_file', help='Path to the JSONL metrics file')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Update interval in seconds (default: 2.0)')
    parser.add_argument('--text', action='store_true',
                        help='Text-mode output (for headless/SSH, auto-refreshes)')
    parser.add_argument('--once', action='store_true',
                        help='Print status once and exit (no live updates)')
    args = parser.parse_args()

    if not os.path.exists(args.metrics_file):
        print(f"Waiting for metrics file: {args.metrics_file}")
        while not os.path.exists(args.metrics_file):
            time.sleep(1)

    # Text mode: print to terminal, no GUI needed
    if args.text or args.once:
        if args.once:
            _print_status(args.metrics_file)
            return
        try:
            while True:
                os.system('clear' if os.name != 'nt' else 'cls')
                _print_status(args.metrics_file)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            return

    # GUI mode
    fig, axes = _setup_figure()
    state = {}

    ani = FuncAnimation(fig, _update, fargs=(args.metrics_file, axes, state),
                        interval=int(args.interval * 1000), cache_frame_data=False)
    plt.show()


if __name__ == '__main__':
    main()
