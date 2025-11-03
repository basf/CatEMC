import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
import numpy as np
from ase.io import read, write
from ase.units import kB
from itertools import combinations
from collections import defaultdict
import pickle


matplotlib.rcParams.update(matplotlib.rcParamsDefault)
from matplotlib import font_manager
font_path = '/mnt/nfs/source/arial.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'

system_name = 'Co0.25Mo0.45Fe0.1Ni0.1Cu0.1NH_111'
target_temperatures = [573, 673, 773, 800, 873]
E_N2 = -16.293 + 0.155  # Etot incl. ZPE RPBE 
E_H2 = -6.984 + 0.269   # Etot incl. ZPE RPBE
E_NH3 = -19.502 + 0.906 # Etot incl. ZPE RPBE
dmu_N = np.arange(-1., 0.001, 0.025)
dmu_H = np.arange(-1., 0.001, 0.025)
grid = np.array(np.meshgrid(dmu_N, dmu_H, target_temperatures)).T.reshape(-1, 3)

T = 800
early_dmu_N = -8.227 - E_N2/2
early_dmu_H = -4.065 - E_H2/2
equil_dmu_N = -8.993 - E_N2/2
equil_dmu_H = -3.985 - E_H2/2
cond1 = np.array([early_dmu_N, early_dmu_H, T])
cond2 = np.array([equil_dmu_N, equil_dmu_H, T])
grid = np.vstack([grid, cond1, cond2])

with open(f'{system_name}_mbar_expectations.pkl', 'rb') as handle:
    expectations = pickle.load(handle)
metal_species = ['Co','Mo','Fe','Ni','Cu']
metal_composition = {'Co': 0.25, 'Mo': 0.45, 'Fe': 0.1, 'Ni': 0.1, 'Cu': 0.1}

fig, axes = plt.subplots(1, 5, figsize=(12.3, 2.3), gridspec_kw = {'wspace':0, 'hspace':0})
axes_flat = axes.flatten()
vmin, vmax = 0., 5.
levels = np.linspace(vmin, vmax, 6)
for i, ax in enumerate(axes_flat):
    m = metal_species[i]
    x_surf_avg = expectations[f'x_surf_{m}'] / metal_composition[m]
    Z = np.empty((len(dmu_H), len(dmu_N)), dtype=float)
    Z.fill(np.nan)
    for iy, muH in enumerate(dmu_H):
        for ix, muN in enumerate(dmu_N):
            idx = np.where(np.isclose(grid[:,0], muN) & np.isclose(grid[:,1], muH) & np.isclose(grid[:,-1], T))[0]
            assert idx.size == 1
            Z[iy, ix] = x_surf_avg[idx[0]]
    Z[Z==0] = np.nan
    cmap = matplotlib.cm.turbo
    cmap.set_bad('lightgray')
    X = dmu_N
    Y = dmu_H
    cf = ax.contourf(X, Y, Z, levels=np.arange(vmin,vmax,1e-3), cmap=cmap, extend='max')
    cf.set_rasterized(True)
    ax.set_title(f'{m}', fontsize=12, y=0.8)
    ax.set_xlabel(r'$\Delta \mu_{\rm{N}}$ / eV', fontsize=10)
    if i in [0]:
        ax.set_ylabel(r'$\Delta \mu_{\rm{H}}$ / eV', fontsize=10)
    if i in [4]:
        ax.set_xlim([X.min(), X.max()])
    else:
        ax.set_xlim([X.min(), X.max()-0.001])
    ax.set_ylim([Y.min(), Y.max()])
    if i not in [0]:
        ax.set_yticklabels([])
    ax.set_xticks([-1., -0.8, -0.6, -0.4, -0.2])
    ax.scatter(early_dmu_N, early_dmu_H, marker='o', c='gold', s=50, alpha=1., edgecolors='black', zorder=3, clip_on=False)
    ax.scatter(equil_dmu_N, equil_dmu_H, marker='*', c='gold', s=100, alpha=1., edgecolors='black', zorder=3, clip_on=False)
cbar = fig.colorbar(cf, ax=axes_flat.tolist(), orientation='vertical', fraction=0.02, pad=0.02)
cbar.ax.set_yticks([0., 1., 2., 3., 4., 5.])
cbar.set_label(r'Average $c_m^{\rm{surf}} / c_m^{\rm{bulk}}$', fontsize=12)

plt.savefig(f'{system_name}_metals_spd_muN_vs_muH.png', bbox_inches='tight', dpi=600)
plt.savefig(f'{system_name}_metals_spd_muN_vs_muH.svg', bbox_inches='tight', dpi=600, format='svg')
plt.clf()
