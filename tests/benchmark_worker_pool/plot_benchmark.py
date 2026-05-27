#!/usr/bin/env python
"""Generate benchmark infographics: Worker Pool vs Subprocess performance."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Load real benchmark data
with open('/mnt/nfs/source/CatEMC_sandip/CatEMC/tests/benchmark_worker_pool/sweep_results.json') as f:
    data = json.load(f)

# Extract metadata
mace_meta = data['mace']['metadata']
pet_meta = data['pet']['metadata']
nequip_meta = data['nequip']['metadata']

# ============ Figure 1: Main infographic (2-panel) ============
fig = plt.figure(figsize=(16, 10), facecolor='white')
fig.suptitle('CatEMC Worker Pool: Persistent Workers vs Subprocess Spawning',
             fontsize=16, fontweight='bold', y=0.97)
fig.text(0.5, 0.935, 'Benchmark: Cu$_7$Ru$_{65}$O$_{144}$ (216 atoms), 3 replicas, 50 relaxation steps, NVIDIA A100-40GB',
         ha='center', fontsize=10, color='#555555')

# --- Panel 1: Speedup vs Rounds (measured data) ---
ax1 = fig.add_subplot(121)

models = ['mace', 'pet', 'nequip']
colors = {'mace': '#2196F3', 'pet': '#FF9800', 'nequip': '#4CAF50'}
labels = {'mace': 'MACE', 'pet': 'PET (Metatomic)', 'nequip': 'NequIP'}
markers = {'mace': 'o', 'pet': 's', 'nequip': '^'}

for model in models:
    rounds = sorted([int(k) for k in data[model].keys() if k != 'metadata'])
    speedups = [data[model][str(r)]['speedup'] for r in rounds]
    ax1.plot(rounds, speedups, f'-{markers[model]}', color=colors[model],
             label=labels[model], markersize=8, linewidth=2.5)

ax1.set_xlabel('EMC Generations (rounds)', fontsize=12)
ax1.set_ylabel('Speedup (×)', fontsize=12)
ax1.set_title('Measured Speedup vs. Number of Generations', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 65)
ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax1.text(62, 1.3, 'no benefit', fontsize=8, color='gray', ha='right')

# Annotate peak values
for model in models:
    rounds = sorted([int(k) for k in data[model].keys() if k != 'metadata'])
    max_r = rounds[-1]
    max_s = data[model][str(max_r)]['speedup']
    ax1.annotate(f'{max_s:.1f}×', xy=(max_r, max_s),
                 xytext=(max_r - 3, max_s + 1.2),
                 fontsize=10, fontweight='bold', color=colors[model],
                 ha='center')

# --- Panel 2: Projected wall-clock time for real EMC runs ---
ax2 = fig.add_subplot(122)

gen_counts = [15, 60, 300, 500, 1500]
bar_width = 0.13
x = np.arange(len(gen_counts))

for i, model in enumerate(models):
    meta = data[model]['metadata']
    wp_startup = meta['wp_startup']
    wp_avg = meta['wp_avg_round_excl_first']
    sp_avg = meta['sp_avg_round']

    wp_times_hr = [(wp_startup + n * wp_avg) / 3600 for n in gen_counts]
    sp_times_hr = [(n * sp_avg) / 3600 for n in gen_counts]

    ax2.bar(x + i * bar_width - 0.2, sp_times_hr, bar_width,
            color=colors[model], alpha=0.35, edgecolor=colors[model], linewidth=1.2)
    ax2.bar(x + i * bar_width - 0.2, wp_times_hr, bar_width,
            color=colors[model], alpha=0.9, edgecolor='white', linewidth=0.5)

# Custom legend
sp_patch = mpatches.Patch(facecolor='gray', alpha=0.35, edgecolor='gray', label='Subprocess (old)')
wp_patch = mpatches.Patch(facecolor='gray', alpha=0.9, edgecolor='white', label='Worker Pool (new)')
model_patches = [mpatches.Patch(facecolor=colors[m], alpha=0.7, label=labels[m]) for m in models]
ax2.legend(handles=[sp_patch, wp_patch] + model_patches, fontsize=9, loc='upper left', ncol=2)

ax2.set_xticks(x)
ax2.set_xticklabels([f'{n} gen' for n in gen_counts], fontsize=10)
ax2.set_ylabel('Wall-clock time (hours)', fontsize=12)
ax2.set_title('Projected Total Runtime for EMC Campaigns', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add speedup annotations on bars
for i, model in enumerate(models):
    meta = data[model]['metadata']
    wp_startup = meta['wp_startup']
    wp_avg = meta['wp_avg_round_excl_first']
    sp_avg = meta['sp_avg_round']
    for j, n in enumerate(gen_counts):
        sp_t = n * sp_avg / 3600
        wp_t = (wp_startup + n * wp_avg) / 3600
        speedup = sp_t / wp_t
        if speedup > 2.5:  # Only annotate significant speedups
            ax2.text(x[j] + i * bar_width - 0.2, sp_t + 0.02,
                     f'{speedup:.0f}×', fontsize=7, ha='center',
                     color=colors[model], fontweight='bold')

plt.tight_layout(rect=[0, 0.08, 1, 0.92])

# Bottom annotation
fig.text(0.5, 0.02,
         'Worker Pool eliminates per-generation subprocess overhead (Python + PyTorch + CUDA bootstrap ≈ 6–14s per call).\n'
         'Model loaded once at startup; subsequent generations execute only the computation.',
         ha='center', fontsize=9, color='#333333',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f7ff', edgecolor='#ccc'))

plt.savefig('/mnt/nfs/source/CatEMC_sandip/CatEMC/images/worker_pool_benchmark.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: images/worker_pool_benchmark.png")


# ============ Figure 2: Per-round breakdown (stacked bar) ============
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
fig2.suptitle('What Happens Each Generation: Compute vs. Overhead', fontsize=14, fontweight='bold')

# Left: per-round time breakdown
model_names = ['MACE', 'PET', 'NequIP']
wp_compute = [mace_meta['wp_avg_round_excl_first'], pet_meta['wp_avg_round_excl_first'], nequip_meta['wp_avg_round_excl_first']]
sp_overhead = [mace_meta['sp_avg_round'] - mace_meta['wp_avg_round_excl_first'],
               pet_meta['sp_avg_round'] - pet_meta['wp_avg_round_excl_first'],
               nequip_meta['sp_avg_round'] - nequip_meta['wp_avg_round_excl_first']]

x3 = np.arange(len(model_names))
bars1 = ax3.bar(x3, wp_compute, 0.5, label='Actual computation', color='#4CAF50', alpha=0.85)
bars2 = ax3.bar(x3, sp_overhead, 0.5, bottom=wp_compute, label='Subprocess overhead\n(eliminated by Worker Pool)',
                color='#f44336', alpha=0.6, hatch='//')

ax3.set_xticks(x3)
ax3.set_xticklabels(model_names, fontsize=12)
ax3.set_ylabel('Time per generation (seconds)', fontsize=11)
ax3.set_title('Time Breakdown per Generation', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10, loc='upper right')
ax3.grid(True, alpha=0.3, axis='y')

# Add percentage labels
for i in range(3):
    total = wp_compute[i] + sp_overhead[i]
    pct = sp_overhead[i] / total * 100
    ax3.text(i, total + 0.3, f'{pct:.0f}% overhead', ha='center', fontsize=10, color='#d32f2f', fontweight='bold')

# Right: projected savings table as text
ax4.axis('off')
ax4.set_title('Projected Time Savings', fontsize=12, fontweight='bold', pad=20)

def format_time(hours):
    """Format time smartly: use hours if >= 0.1h, minutes if >= 1min, else seconds."""
    if hours >= 0.1:
        return f"{hours:.1f}h"
    elif hours * 60 >= 1:
        return f"{hours*60:.1f}min"
    else:
        return f"{hours*3600:.0f}s"

def format_time_cell(sp_hr, wp_hr, speedup, saved_hr):
    sp_str = format_time(sp_hr)
    wp_str = format_time(wp_hr)
    saved_str = format_time(saved_hr)
    return f"{sp_str} → {wp_str}\n({speedup:.0f}× faster, saves {saved_str})"

table_data = []
headers = ['Model', '300 gen', '500 gen', '1500 gen']
for model, name in zip(models, ['MACE', 'PET', 'NequIP']):
    meta = data[model]['metadata']
    wp_startup = meta['wp_startup']
    wp_avg = meta['wp_avg_round_excl_first']
    sp_avg = meta['sp_avg_round']
    row = [name]
    for n in [300, 500, 1500]:
        sp_hr = n * sp_avg / 3600
        wp_hr = (wp_startup + n * wp_avg) / 3600
        saved_hr = sp_hr - wp_hr
        speedup = sp_hr / wp_hr
        row.append(format_time_cell(sp_hr, wp_hr, speedup, saved_hr))
    table_data.append(row)

table = ax4.table(cellText=table_data, colLabels=headers, loc='center',
                  cellLoc='center', colWidths=[0.12, 0.29, 0.29, 0.29])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 2.2)

# Style the table
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2196F3')
        cell.set_text_props(color='white', fontweight='bold')
    elif col == 0:
        cell.set_facecolor('#f5f5f5')
        cell.set_text_props(fontweight='bold')
    else:
        cell.set_facecolor('#fafafa')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('/mnt/nfs/source/CatEMC_sandip/CatEMC/images/worker_pool_breakdown.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: images/worker_pool_breakdown.png")
print("\nDone!")
