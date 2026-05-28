"""Plot per-generation time for CephFS vs scratch — 1, 6, 24, and 48 workers + speedup scaling."""
import matplotlib.pyplot as plt
import numpy as np

# === 1 WORKER DATA ===
cephfs_1w = [7.0, 7.7, 8.5, 9.2, 10.1, 10.7, 11.0, 11.5, 12.4, 67.6,
             14.7, 15.1, 15.9, 16.9, 22.1, 22.6, 22.8, 23.8, 23.2]
scratch_1w = [6.2, 6.5, 6.6, 6.8, 6.8, 6.7, 6.9, 7.1, 7.3,
              7.4, 7.3, 7.5, 7.4, 7.5, 7.6, 7.5, 7.6, 7.7, 7.8]

# === 6 WORKER DATA ===
cephfs_6w = [2.3, 4.6, 11.0, 12.5, 11.7, 11.6, 11.6, 10.9, 10.5, 23.5,
             9.1, 9.7, 10.2, 11.0, 11.7, 12.4, 13.7, 17.9, 18.7]
scratch_6w = [1.3, 1.3, 1.4, 1.5, 1.5, 1.6, 1.6, 1.7, 1.7, 10.6,
              1.7, 1.9, 1.9, 2.0, 2.0, 2.2, 2.2, 2.2, 2.4]

# === 24 WORKER DATA ===
cephfs_24w = [1.7, 2.5, 3.0, 3.6, 4.3, 4.7, 5.4, 6.0, 6.9, 17.7,
              8.5, 9.0, 9.6, 10.4, 11.1, 11.9, 12.3, 12.6, 13.4]
scratch_24w = [0.5, 0.6, 0.7, 0.7, 0.8, 0.8, 0.8, 1.0, 1.0, 4.6,
               1.1, 1.2, 1.1, 1.3, 1.3, 1.3, 1.4, 1.4, 1.5]

# === 48 WORKER DATA ===
cephfs_48w = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.9, 6.7, 17.2,
              8.6, 9.1, 10.4, 10.6, 10.9, 11.8, 12.3, 12.7, 13.2]
scratch_48w = [0.7, 0.7, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 1.0, 4.0,
               1.1, 1.2, 1.1, 1.2, 1.4, 1.3, 1.3, 1.4, 1.4]

generations = np.arange(2, 2 + 19)

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

# --- Top row: per-generation time series (4 panels) ---
configs = [
    (cephfs_1w, scratch_1w, '1 Worker'),
    (cephfs_6w, scratch_6w, '6 Workers'),
    (cephfs_24w, scratch_24w, '24 Workers'),
    (cephfs_48w, scratch_48w, '48 Workers'),
]

for i, (cephfs, scratch, title) in enumerate(configs):
    ax = fig.add_subplot(gs[0, i])
    ax.plot(generations, cephfs, 'o-', color='#d62728', linewidth=2, markersize=4, label='CephFS')
    ax.plot(generations, scratch, 's-', color='#2ca02c', linewidth=2, markersize=4, label='/mnt/scratch')
    if cephfs[9] > 15:
        ax.annotate(f'MD\n({cephfs[9]:.0f}s)', xy=(generations[9], cephfs[9]),
                    xytext=(generations[9]+1.5, cephfs[9]-5), fontsize=8, color='#d62728',
                    arrowprops=dict(arrowstyle='->', color='#d62728'))
    ax.set_xlabel('EMC Generation', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 75)
    if i == 0:
        ax.set_ylabel('Time per Generation (s)', fontsize=10)

# --- Bottom row: summary panels (use middle 3 of 4 columns, centered) ---
# Bottom left: bar chart
ax_bar = fig.add_subplot(gs[1, 0:2])
workers = [1, 6, 24, 48]
cephfs_avg = [23.7, 14.1, 9.8, 9.8]
scratch_avg = [15.6, 3.2, 1.6, 1.5]

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

# Bottom center-right: speedup scaling
ax_speed = fig.add_subplot(gs[1, 2])
workers_arr = np.array([1, 6, 24, 48])
cephfs_speedup = [cephfs_avg[0] / c for c in cephfs_avg]
scratch_speedup = [scratch_avg[0] / s for s in scratch_avg]
ideal_speedup = workers_arr / workers_arr[0]

ax_speed.plot(workers_arr, cephfs_speedup, 'o-', color='#d62728', linewidth=2.5, markersize=8, label='CephFS')
ax_speed.plot(workers_arr, scratch_speedup, 's-', color='#2ca02c', linewidth=2.5, markersize=8, label='/mnt/scratch')
ax_speed.plot(workers_arr, ideal_speedup, 'k--', linewidth=1.5, alpha=0.5, label='Ideal linear')
ax_speed.set_xlabel('Workers', fontsize=10)
ax_speed.set_ylabel('Speedup vs 1 worker', fontsize=10)
ax_speed.set_title('Scaling Efficiency', fontsize=11, fontweight='bold')
ax_speed.legend(fontsize=9, loc='upper left')
ax_speed.grid(True, alpha=0.3)
ax_speed.set_xticks(workers_arr)
for i, (cs, ss) in enumerate(zip(cephfs_speedup, scratch_speedup)):
    ax_speed.annotate(f'{cs:.1f}×', xy=(workers_arr[i], cs), xytext=(5, -12),
                      textcoords='offset points', fontsize=9, color='#d62728', fontweight='bold')
    ax_speed.annotate(f'{ss:.1f}×', xy=(workers_arr[i], ss), xytext=(5, 5),
                      textcoords='offset points', fontsize=9, color='#2ca02c', fontweight='bold')

# Bottom right: CephFS overhead factor
ax_overhead = fig.add_subplot(gs[1, 3])
overhead = [c / s for c, s in zip(cephfs_avg, scratch_avg)]
ax_overhead.bar(x, overhead, color='#ff7f0e', alpha=0.8, width=0.5)
ax_overhead.set_xlabel('Workers', fontsize=10)
ax_overhead.set_ylabel('CephFS / Scratch ratio', fontsize=10)
ax_overhead.set_title('CephFS I/O Overhead Factor', fontsize=11, fontweight='bold')
ax_overhead.set_xticks(x)
ax_overhead.set_xticklabels(workers)
ax_overhead.grid(True, alpha=0.3, axis='y')
ax_overhead.axhline(y=1, color='k', linestyle='--', alpha=0.3)
for i, v in enumerate(overhead):
    ax_overhead.text(i, v + 0.1, f'{v:.1f}×', ha='center', fontsize=11, fontweight='bold')

fig.suptitle('CephFS I/O Bottleneck Analysis — EMC Worker Scaling on H100\n'
             '(48 replicas, traj_write_interval=1, NequIP model)',
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig('/fs/cephfs/data/qm_inorganics/sources/CatEMC_sandip/CatEMC/tests/benchmark_io_reduction/fs_latency_comparison.png',
            dpi=150, bbox_inches='tight')
print("Saved: fs_latency_comparison.png")
plt.close('all')
