Visualize results
=================

This page contains the plotting scripts from the README to reproduce surface
phase diagrams: metal segregation heatmaps and adsorbate coverage diagrams.
Both scripts use Matplotlib (no seaborn style presets assumed); adjust font
paths and plotting parameters to your local environment.

.. contents::
   :local:
   :depth: 2

Metal segregation surface phase diagram
--------------------------------------

The script below takes GC-EMC analysis outputs (``*_mbar_expectations.pkl``)
and plots average surface metal concentrations normalized by bulk.

.. code-block:: python

   import matplotlib
   import matplotlib.pyplot as plt
   import numpy as np
   from ase.io import read
   import pickle

   matplotlib.rcParams.update(matplotlib.rcParamsDefault)
   # Optional: set a system font if available
   # from matplotlib import font_manager
   # font_path = '/mnt/nfs/source/arial.ttf'
   # font_manager.fontManager.addfont(font_path)
   # prop = font_manager.FontProperties(fname=font_path)
   # plt.rcParams['font.family'] = 'sans-serif'
   # plt.rcParams['font.sans-serif'] = prop.get_name()

   system_name = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5OH_311'
   sim_temperatures = [1023,]
   T = 1023
   dmu_O = np.arange(-5., -2.999, 0.02)
   dmu_H = np.arange(-1., 0.001, 0.01)
   grid = np.array(np.meshgrid(dmu_O, dmu_H, sim_temperatures)).T.reshape(-1, 3)

   with open(f'{system_name}_mbar_expectations.pkl', 'rb') as handle:
       expectations = pickle.load(handle)

   metal_species = ['Co', 'Ni', 'Mg', 'Zn', 'Mn', 'Al']
   metal_composition = {'Co':0.2/3,'Ni':0.2/3,'Mg':0.2/3,'Zn':0.2/3,'Mn':0.2/3,'Al':2/3}

   # Plot grid: 2x3 subplots
   fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.5), gridspec_kw={'wspace':0, 'hspace':0})
   axes_flat = axes.flatten()
   vmin, vmax = 0., 5.

   for i, ax in enumerate(axes_flat):
       m = metal_species[i]
       x_surf_avg = expectations[f'x_surf_{m}'] / metal_composition[m]

       # Construct Z with shape (Ny, Nx) where rows -> dmu_H, cols -> dmu_O
       Z = np.empty((len(dmu_O), len(dmu_H)), dtype=float)
       Z.fill(np.nan)
       for iy, muH in enumerate(dmu_H):
           for ix, muO in enumerate(dmu_O):
               idx = np.where(np.isclose(grid[:,0], muO) & np.isclose(grid[:,1], muH) & np.isclose(grid[:,-1], T))[0]
               if idx.size == 1:
                   Z[iy, ix] = x_surf_avg[idx[0]]
               else:
                   Z[iy, ix] = np.nan
       Z[Z==0] = np.nan

       cmap = matplotlib.cm.turbo
       cmap.set_bad('lightgray')
       cf = ax.contourf(dmu_O, dmu_H, Z, levels=np.arange(vmin, vmax, 1e-3), cmap=cmap, extend='max')
       cf.set_rasterized(True)
       ax.set_title(f'{m}', fontsize=12, y=0.8)

       if i in [3, 4, 5]:
           ax.set_xlabel(r'$\Delta \mu_{\rm{O}}$ / eV', fontsize=10)
       if i in [0, 3]:
           ax.set_ylabel(r'$\Delta \mu_{\rm{H}}$ / eV', fontsize=10)
       if i in [2, 5]:
           ax.set_xlim([-5, -3])
       else:
           ax.set_xlim([-5, -3.001])
       if i in [3, 4, 5]:
           ax.set_ylim([-1., -0.001])
       else:
           ax.set_ylim([-1, 0.])
       if i in [0, 1, 2]:
           ax.set_xticklabels([])
       if i not in [0, 3]:
           ax.set_yticklabels([])

   cbar = fig.colorbar(cf, ax=axes_flat.tolist(), orientation='vertical', fraction=0.02, pad=0.02)
   cbar.ax.set_yticks([0., 1., 2., 3., 4., 5.])
   cbar.set_label(r'Average $c_m^{\rm{surf}} / c_m^{\rm{bulk}}$', fontsize=12)

   plt.show()

Adsorbate coverage surface phase diagram
---------------------------------------

Plot adsorbate coverage maps (N/H/combined) similar to the metal segregation plots.

.. code-block:: python

   import matplotlib
   import matplotlib.pyplot as plt
   import numpy as np
   import pickle

   matplotlib.rcParams.update(matplotlib.rcParamsDefault)
   # optional font setup as above

   system_name = 'Co0.25Mo0.45Fe0.1Ni0.1Cu0.1NH_111'
   sim_temperatures = [573, 673, 773, 873]
   T = 773
   dmu_N = np.arange(-1., 0.001, 0.01)
   dmu_H = np.arange(-1., 0.001, 0.01)
   grid = np.array(np.meshgrid(dmu_N, dmu_H, sim_temperatures)).T.reshape(-1, 3)

   with open(f'{system_name}_mbar_expectations.pkl', 'rb') as handle:
       expectations = pickle.load(handle)

   adsorbate_species = ['N', 'H', 'NH', 'NH2', 'all']

   fig, axes = plt.subplots(1, 5, figsize=(12.3, 2.3), gridspec_kw={'wspace':0, 'hspace':0})
   axes_flat = axes.flatten()
   vmin, vmax = 0., 1.

   for i, ax in enumerate(axes_flat):
       ads = adsorbate_species[i]
       if ads == 'all':
           ads_coverage_avg = expectations[f'ads_coverage']
       else:
           ads_coverage_avg = expectations[f'ads_coverage_{ads}']

       Z = np.empty((len(dmu_N), len(dmu_H)), dtype=float)
       Z.fill(np.nan)
       for iy, muH in enumerate(dmu_H):
           for ix, muN in enumerate(dmu_N):
               idx = np.where(np.isclose(grid[:,0], muN) & np.isclose(grid[:,1], muH) & np.isclose(grid[:,-1], T))[0]
               if idx.size == 1:
                   Z[iy, ix] = ads_coverage_avg[idx[0]]
               else:
                   Z[iy, ix] = np.nan
       Z[Z==0] = np.nan

       cmap = matplotlib.cm.turbo
       cmap.set_bad('lightgray')
       cf = ax.contourf(dmu_N, dmu_H, Z, levels=np.arange(vmin, vmax, 1e-3), cmap=cmap, extend='max')
       cf.set_rasterized(True)

       if ads in ['NH','all','NH2']:
           color = 'k'
       else:
           color = 'w'
       if ads == 'NH2':
           title = r'NH$_2$*'
       elif ads == 'all':
           title = 'All adsorbates'
       else:
           title = f'{ads}*'
       ax.set_title(title, fontsize=12, y=0.8, color=color)
       ax.set_xlabel(r'$\Delta \mu_{\rm{N}}$ / eV', fontsize=10)
       if i == 0:
           ax.set_ylabel(r'$\Delta \mu_{\rm{H}}$ / eV', fontsize=10)
       if i == 4:
           ax.set_xlim([-1, 0.])
       else:
           ax.set_xlim([-1, -0.001])
       ax.set_ylim([-1, 0.])
       if i not in [0]:
           ax.set_yticklabels([])

   cbar = fig.colorbar(cf, ax=axes_flat.tolist(), orientation='vertical', fraction=0.02, pad=0.02)
   cbar.ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
   cbar.set_label('Average coverage / ML', fontsize=12)

   plt.show()

