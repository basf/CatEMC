from ase.io import read, write
from ase.visualize import view
from ase.units import Bohr, kB
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import math
import random
import pickle
from itertools import combinations
from matplotlib.collections import PatchCollection
from matplotlib import collections  as mc
import matplotlib

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# Set font properties
from matplotlib import font_manager
font_path = '/mnt/nfs/source/arial.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams['text.usetex'] = False
plt.rcParams['svg.fonttype'] = 'none'

plt.rcParams['figure.figsize'] = 3, 2.5

system_name = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5COH_311'
cond_id = -1 
template = read(f'template_Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5_311_slab_mace_opt.traj')
A_surf = np.linalg.norm(np.cross(template.cell[0], template.cell[1]))
elements = ['Co','Ni','Mg','Zn','Mn','Al','C','O','H']
all_atom_pairs = list(combinations(elements, 2)) + [(m,m) for m in elements]
with open(f'{system_name}_mbar_expectations.pkl', 'rb') as handle:
    expectations = pickle.load(handle)

corr = np.zeros((len(elements), len(elements)))
print(expectations.keys())
for m1, m2 in all_atom_pairs:
    bond_count_avg = expectations[f'{m1}-{m2}'][cond_id]
    corr[elements.index(m2), elements.index(m1)] = bond_count_avg / A_surf
print(corr)
# Create a mask
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
cmap = matplotlib.cm.turbo
cmap.set_bad('white')
ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=0., vmax=0.2, annot=True, 
                 annot_kws={"fontsize": 7}, cbar_kws={'shrink': 0.95}, 
                 fmt='0.2f', square=True, linewidth=1, linecolor='w')
ax.set_xticklabels(elements, fontsize=8)
ax.set_yticklabels(elements, fontsize=8)
cbar = ax.collections[0].colorbar
ticks = np.linspace(0.0, 0.2, 6) 
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{t:.2f}' for t in ticks])
cbar.set_label(r'Average surface bond count per $\rm{\AA}^2$', size=8)
cbar.ax.tick_params(labelsize=8)

plt.savefig(f'{system_name}_surf_bond_matrix.png', bbox_inches='tight', format='png', dpi=600)
plt.savefig(f'{system_name}_surf_bond_matrix.svg', bbox_inches='tight', format='svg', dpi=600)
plt.clf()
