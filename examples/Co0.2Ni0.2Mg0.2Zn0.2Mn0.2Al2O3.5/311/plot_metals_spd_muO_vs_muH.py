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
metal_species = ['Co', 'Ni', 'Mg', 'Zn', 'Mn', 'Al']
metal_composition = {'Co':0.2/3,'Ni':0.2/3,'Mg':0.2/3,'Zn':0.2/3,'Mn':0.2/3,'Al':2/3}

fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.5), gridspec_kw = {'wspace':0, 'hspace':0})
axes_flat = axes.flatten()

vmin, vmax = 0., 5.
levels = np.linspace(vmin, vmax, 6)
for i, ax in enumerate(axes_flat):
    m = metal_species[i]
    x_surf_avg = expectations[f'x_surf_{m}'] / metal_composition[m]
    Z = np.empty((len(dmu_H), len(dmu_O)), dtype=float)
    Z.fill(np.nan)
    for iy, muH in enumerate(dmu_H):
        for ix, muO in enumerate(dmu_O):
            idx = np.where(np.isclose(grid[:,0], muO) & np.isclose(grid[:,1], muH))[0]
            assert idx.size == 1
            Z[iy, ix] = x_surf_avg[idx[0]]
    Z[Z==0] = np.nan
    cmap = matplotlib.cm.turbo
    cmap.set_bad('lightgray')
    X = dmu_O
    Y = dmu_H
    cf = ax.contourf(X, Y, Z, levels=np.arange(vmin,vmax,1e-3), cmap=cmap, extend='max')
    cf.set_rasterized(True)
    ax.set_title(f'{m}', fontsize=12, y=0.8)
    if i in [3, 4, 5]:
        ax.set_xlabel(r'$\Delta \mu_{\rm{O}}$ / eV', fontsize=10)
    if i in [0, 3]:
        ax.set_ylabel(r'$\Delta \mu_{\rm{H}}$ / eV', fontsize=10)
    if i in [2, 5]:
        ax.set_xlim([X.min(), X.max()])
    else:
        ax.set_xlim([X.min(), X.max()-0.001])
    if i in [3, 4, 5]:
        ax.set_ylim([Y.min(), Y.max()-0.001])
    else:
        ax.set_ylim([Y.min(), Y.max()])
    if i in [0, 1, 2]:
        ax.set_xticklabels([])
    if i not in [0, 3]:
        ax.set_yticklabels([])
    ax.set_xticks([-4.8, -4.4, -4., -3.6, -3.2])
    ax.scatter(equil_dmu_O, equil_dmu_H, marker='*', c='gold', s=100, alpha=1., edgecolors='black', zorder=3)
cbar = fig.colorbar(cf, ax=axes_flat.tolist(), orientation='vertical', fraction=0.02, pad=0.02)
cbar.ax.set_yticks([0., 1., 2., 3., 4., 5.])
cbar.set_label(r'Average $c_m^{\rm{surf}} / c_m^{\rm{bulk}}$', fontsize=12)

plt.savefig(f'{system_name}_metals_spd_muO_vs_muH.png', bbox_inches='tight', dpi=600)
plt.savefig(f'{system_name}_metals_spd_muO_vs_muH.svg', bbox_inches='tight', dpi=600, format='svg')
plt.clf()
