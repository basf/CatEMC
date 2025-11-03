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

system_name = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5OH_311'
sim_temperatures = [1023]
T = 1023
E_O2 = -8.768 + 0.101   # Etot incl. ZPE PBE
E_H2 = -6.773 + 0.267   # Etot incl. ZPE PBE
E_H2O = -14.237 + 0.572 # Etot incl. ZPE PBE
equil_dmu_O = -8.8209 - E_O2 / 2
equil_dmu_H = -3.9805 - E_H2 / 2
dmu_O = np.arange(-5., -2.999, 0.02)
dmu_H = np.arange(-1., 0.001, 0.01)
grid = np.array(np.meshgrid(dmu_O, dmu_H)).T.reshape(-1, 2)

with open(f'{system_name}_mbar_expectations.pkl', 'rb') as handle:
    expectations = pickle.load(handle)
adsorbate_species = ['O', 'H', 'OH', 'all']

fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.6), gridspec_kw = {'wspace':0, 'hspace':0})
axes_flat = axes.flatten()

vmin, vmax = 0., 1.25
levels = np.linspace(vmin, vmax, 6)
for i, ax in enumerate(axes_flat):
    ads = adsorbate_species[i] 
    if ads == 'all':
        ads_coverage_avg = expectations[f'ads_coverage']
    else:
        ads_coverage_avg = expectations[f'ads_coverage_{ads}']
    Z = np.empty((len(dmu_H), len(dmu_O)), dtype=float)
    Z.fill(np.nan)
    for iy, muH in enumerate(dmu_H):
        for ix, muO in enumerate(dmu_O):
            idx = np.where(np.isclose(grid[:,0], muO) & np.isclose(grid[:,1], muH))[0]
            assert idx.size == 1
            Z[iy, ix] = ads_coverage_avg[idx[0]]
    Z[Z==0] = np.nan
    cmap = matplotlib.cm.turbo
    cmap.set_bad('lightgray')
    cmap.set_over('maroon')
    X = dmu_O
    Y = dmu_H
    cf = ax.contourf(X, Y, Z, levels=np.arange(vmin,vmax,1e-3), cmap=cmap, extend='max')
    cf.set_rasterized(True)

    if ads == 'H':
        color = 'w'
    else:
        color = 'k'
    if ads == 'all':
        ads = 'All adsorbates'
    else:
        ads += '*'
    ax.set_title(ads, fontsize=12, y=0.8, color=color)
    ax.set_xlabel(r'$\Delta \mu_{\rm{O}}$ / eV', fontsize=10)
    if i in [0,2]:
        ax.set_ylabel(r'$\Delta \mu_{\rm{H}}$ / eV', fontsize=10)
    if i in [1,3]:
        ax.set_xlim([X.min(), X.max()])
    else:
        ax.set_xlim([X.min(), X.max()-0.001])
    if i in [0,1]:
        ax.set_ylim([Y.min(), Y.max()])
    else:
        ax.set_ylim([Y.min(), Y.max()-0.001])
    if i in [0,1]:
        ax.set_xticklabels([])
    if i in [1,3]:
        ax.set_yticklabels([])
    ax.set_xticks([-4.8, -4.4, -4., -3.6, -3.2])
    ax.scatter(equil_dmu_O, equil_dmu_H, marker='*', c='gold', s=100, alpha=1., edgecolors='black', zorder=3)
cbar = fig.colorbar(cf, ax=axes_flat.tolist(), orientation='vertical', fraction=0.02, pad=0.02)
cbar.ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00, 1.25])
cbar.set_label('Average coverage / ML', fontsize=12)

plt.savefig(f'{system_name}_adsorbates_spd_muO_vs_muH.png', bbox_inches='tight', dpi=600)
plt.savefig(f'{system_name}_adsorbates_spd_muO_vs_muH.svg', bbox_inches='tight', dpi=600, format='svg')
plt.clf()
