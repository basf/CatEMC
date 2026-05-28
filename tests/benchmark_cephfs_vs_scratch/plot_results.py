"""Plot CephFS vs Scratch benchmark results.

Reads results.json (populated by run_benchmark.py) and generates
a comprehensive figure showing:
  - Per-generation time series (from .out files)
  - Average performance bar chart
  - Speedup scaling comparison
  - CephFS overhead factor

Usage: python plot_results.py
"""
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'results.json')


def parse_per_gen_times(out_file):
    """Extract per-generation elapsed times from benchmark output file."""
    cephfs_times = []
    scratch_times = []
    current = None
    
    with open(out_file) as f:
        for line in f:
            if 'Test 1: CephFS' in line:
                current = 'cephfs'
            elif 'Test 2: /mnt/scratch' in line:
                current = 'scratch'
            elif 'Creating and evaluating generation' in line:
                m = re.search(r'elapsed: ([\d.]+)s', line)
                if m:
                    t = float(m.group(1))
                    if t > 0:  # skip gen 1 (elapsed=0)
                        if current == 'cephfs':
                            cephfs_times.append(t)
                        elif current == 'scratch':
                            scratch_times.append(t)
    return cephfs_times, scratch_times


def main():
    # Load summary results
    if not os.path.exists(RESULTS_FILE):
        print(f"No results.json found. Run benchmarks first with run_all.sh")
        return
    
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    
    results.sort(key=lambda r: r['num_workers'])
    workers = [r['num_workers'] for r in results]
    cephfs_avg = [r['cephfs_per_gen_s'] for r in results]
    scratch_avg = [r['scratch_per_gen_s'] for r in results]
    
    # Try to load per-gen data from .out files
    per_gen_data = {}
    for nw in workers:
        out_file = os.path.join(SCRIPT_DIR, f'benchmark_{nw}w.out')
        if os.path.exists(out_file):
            c, s = parse_per_gen_times(out_file)
            if c and s:
                per_gen_data[nw] = (c, s)
    
    # --- Create figure ---
    n_panels = len(per_gen_data)
    fig = plt.figure(figsize=(18, 10))
    
    if n_panels > 0:
        gs = fig.add_gridspec(2, max(n_panels, 3), hspace=0.35, wspace=0.3)
        
        # Top row: per-generation time series
        for i, nw in enumerate(sorted(per_gen_data.keys())):
            ax = fig.add_subplot(gs[0, i])
            cephfs_t, scratch_t = per_gen_data[nw]
            n = min(len(cephfs_t), len(scratch_t))
            gens = np.arange(2, 2 + n)
            ax.plot(gens, cephfs_t[:n], 'o-', color='#d62728', linewidth=2, markersize=4, label='CephFS')
            ax.plot(gens, scratch_t[:n], 's-', color='#2ca02c', linewidth=2, markersize=4, label='/mnt/scratch')
            ax.set_xlabel('EMC Generation', fontsize=10)
            ax.set_title(f'{nw} Workers', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_ylabel('Time per Generation (s)', fontsize=10)
            # Set y-limit based on max value
            ymax = max(max(cephfs_t[:n]), max(scratch_t[:n])) * 1.2
            ax.set_ylim(0, min(ymax, 80))
    else:
        gs = fig.add_gridspec(1, 3, hspace=0.35, wspace=0.3)
    
    # Bottom row (or only row): summary panels
    row = 1 if n_panels > 0 else 0
    
    # Bar chart
    ax_bar = fig.add_subplot(gs[row, 0])
    x = np.arange(len(workers))
    width = 0.35
    bars1 = ax_bar.bar(x - width/2, cephfs_avg, width, label='CephFS', color='#d62728', alpha=0.8)
    bars2 = ax_bar.bar(x + width/2, scratch_avg, width, label='/mnt/scratch', color='#2ca02c', alpha=0.8)
    ax_bar.set_xlabel('Workers', fontsize=10)
    ax_bar.set_ylabel('Avg Time per Gen (s)', fontsize=10)
    ax_bar.set_title('Average Performance', fontsize=11, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(workers)
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, alpha=0.3, axis='y')
    ax_bar.bar_label(bars1, fmt='%.1f', padding=2, fontsize=9)
    ax_bar.bar_label(bars2, fmt='%.1f', padding=2, fontsize=9)
    
    # Speedup scaling
    ax_speed = fig.add_subplot(gs[row, 1])
    cephfs_speedup = [cephfs_avg[0] / c for c in cephfs_avg]
    scratch_speedup = [scratch_avg[0] / s for s in scratch_avg]
    workers_arr = np.array(workers)
    ideal = workers_arr / workers_arr[0]
    
    ax_speed.plot(workers_arr, scratch_speedup, 's-', color='#2ca02c', linewidth=2.5, markersize=8, label='/mnt/scratch')
    ax_speed.plot(workers_arr, cephfs_speedup, 'o-', color='#d62728', linewidth=2.5, markersize=8, label='CephFS')
    ax_speed.plot(workers_arr, ideal, 'k--', linewidth=1.5, alpha=0.5, label='Ideal linear')
    ax_speed.set_xlabel('Workers', fontsize=10)
    ax_speed.set_ylabel('Speedup vs 1 worker', fontsize=10)
    ax_speed.set_title('Scaling Efficiency', fontsize=11, fontweight='bold')
    ax_speed.legend(fontsize=9, loc='upper left')
    ax_speed.grid(True, alpha=0.3)
    ax_speed.set_xticks(workers_arr)
    for i, (cs, ss) in enumerate(zip(cephfs_speedup, scratch_speedup)):
        ax_speed.annotate(f'{cs:.1f}×', xy=(workers_arr[i], cs),
                          xytext=(5, -5), textcoords='offset points',
                          fontsize=9, color='#d62728', fontweight='bold')
        ax_speed.annotate(f'{ss:.1f}×', xy=(workers_arr[i], ss),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=9, color='#2ca02c', fontweight='bold')
    
    # Overhead factor
    ax_oh = fig.add_subplot(gs[row, 2])
    overhead = [c / s for c, s in zip(cephfs_avg, scratch_avg)]
    ax_oh.bar(x, overhead, color='#ff7f0e', alpha=0.8, width=0.5)
    ax_oh.set_xlabel('Workers', fontsize=10)
    ax_oh.set_ylabel('CephFS / Scratch ratio', fontsize=10)
    ax_oh.set_title('CephFS I/O Overhead Factor', fontsize=11, fontweight='bold')
    ax_oh.set_xticks(x)
    ax_oh.set_xticklabels(workers)
    ax_oh.grid(True, alpha=0.3, axis='y')
    ax_oh.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    for i, v in enumerate(overhead):
        ax_oh.text(i, v + 0.1, f'{v:.1f}×', ha='center', fontsize=11, fontweight='bold')
    
    fig.suptitle('CephFS I/O Bottleneck Analysis — EMC Worker Scaling on H100\n'
                 '(48 replicas, traj_write_interval=1, NequIP model)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    out_path = os.path.join(SCRIPT_DIR, 'benchmark_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close('all')


if __name__ == '__main__':
    main()
