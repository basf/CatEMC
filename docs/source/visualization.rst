Visualization — Surface phase diagrams
=====================================

This page focuses on plotting surface phase diagrams from GC-EMC MBAR outputs.
We include complete plotting scripts that turn ``*_mbar_expectations.pkl`` into
publication-quality heatmaps for metal segregation and adsorbate coverages.

.. contents::
   :local:
   :depth: 2

Data inputs
-----------

* ``*_mbar_expectations.pkl`` — produced by :class:`catemc.analysis.GCEMCAnalysis`.
* Reaction condition grid arrays (dmu vectors & temperatures).
* Optional: sampled trajectories (``*_mbar.traj``) for overlaying snapshots.

Plot 1 — Metal segregation heatmaps
-----------------------------------

This script constructs 2×3 subplots showing normalized surface metal fraction
relative to bulk composition for each metal species.

.. code-block:: python

   import matplotlib
   import matplotlib.pyplot as plt
   import numpy as np
   import pickle

   matplotlib.rcParams.update(matplotlib.rcParamsDefault)

   system_name = 'Co_surface'
   T = 1023
   dmu_O = np.arange(-5., -3.0, 0.02)
   dmu_H = np.arange(-1., 0.01, 0.01)
   grid = np.array(np.meshgrid(dmu_O, dmu_H, [T])).T.reshape(-1, 3)

   with open(f'{system_name}_mbar_expectations.pkl', 'rb') as handle:
       expectations = pickle.load(handle)

   metal_species = ['Co', 'Ni', 'Mg', 'Zn', 'Mn', 'Al']
   metal_composition = {'Co':0.2/3,'Ni':0.2/3,'Mg':0.2/3,'Zn':0.2/3,'Mn':0.2/3,'Al':2/3}

   fig, axes = plt.subplots(2, 3, figsize=(9, 6), gridspec_kw={'wspace':0.05, 'hspace':0.12})
   axes_flat = axes.flatten()
   vmin, vmax = 0., 5.

   for i, ax in enumerate(axes_flat):
       m = metal_species[i]
       # average surface fraction array for that metal stored under key `x_surf_{m}`
       x_surf_avg = expectations[f'x_surf_{m}'] / metal_composition[m]

       Z = np.full((len(dmu_O), len(dmu_H)), np.nan)
       for iy, muH in enumerate(dmu_H):
           for ix, muO in enumerate(dmu_O):
               idx = np.where(np.isclose(grid[:,0], muO) & np.isclose(grid[:,1], muH) & np.isclose(grid[:,-1], T))[0]
               if idx.size == 1:
                   Z[ix, iy] = x_surf_avg[idx[0]]
       # transposition/axis ordering depends on preferred plotting semantics
       cf = ax.contourf(dmu_O, dmu_H, Z.T, levels=np.linspace(vmin, vmax, 21), extend='max')
       ax.set_title(m)
       if i in [4, 5]:
           ax.set_xlabel(r'$\Delta \mu_{\rm O}$ / eV')
       if i in [0, 3]:
           ax.set_ylabel(r'$\Delta \mu_{\rm H}$ / eV')
       ax.set_xlim([-5, -3])
       ax.set_ylim([-1, 0])

   cbar = fig.colorbar(cf, ax=axes_flat.tolist(), orientation='vertical', fraction=0.02, pad=0.02)
   cbar.set_label('Average $c_m^{\\mathrm{surf}} / c_m^{\\mathrm{bulk}}$')

   plt.tight_layout()
   plt.show()

Plot 2 — Adsorbate coverage maps
-------------------------------

This script plots coverage maps for multiple adsorbate species side-by-side.

.. code-block:: python

   import matplotlib
   import matplotlib.pyplot as plt
   import numpy as np
   import pickle

   matplotlib.rcParams.update(matplotlib.rcParamsDefault)

   system_name = 'Co_surface'
   T = 773
   dmu_N = np.arange(-1.0, 0.01, 0.01)
   dmu_H = np.arange(-1.0, 0.01, 0.01)
   grid = np.array(np.meshgrid(dmu_N, dmu_H, [T])).T.reshape(-1, 3)

   with open(f'{system_name}_mbar_expectations.pkl', 'rb') as handle:
       expectations = pickle.load(handle)

   adsorbate_species = ['N', 'H', 'NH', 'NH2', 'all']
   fig, axes = plt.subplots(1, 5, figsize=(14, 2.6), gridspec_kw={'wspace':0.03})

   for i, ax in enumerate(axes):
       ads = adsorbate_species[i]
       if ads == 'all':
           arr = expectations['ads_coverage']
       else:
           arr = expectations[f'ads_coverage_{ads}']

       Z = np.full((len(dmu_N), len(dmu_H)), np.nan)
       for iy, muH in enumerate(dmu_H):
           for ix, muN in enumerate(dmu_N):
               idx = np.where(np.isclose(grid[:,0], muN) & np.isclose(grid[:,1], muH) & np.isclose(grid[:,-1], T))[0]
               if idx.size == 1:
                   Z[ix, iy] = arr[idx[0]]
       cf = ax.contourf(dmu_N, dmu_H, Z.T, levels=np.linspace(0,1,21), extend='both')
       if i == 0:
           ax.set_ylabel(r'$\Delta \mu_{\rm H}$ / eV')
       ax.set_title(ads)
       ax.set_xlim([-1, 0])
       ax.set_ylim([-1, 0])

   cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02)
   cbar.set_label('Average coverage / ML')

   plt.tight_layout()
   plt.show()

Plot export & tips
------------------

* Use ``cf.set_rasterized(True)`` for large grids to keep vector output sizes small for publication-quality PDF/SVG exports.
* When grid points appear missing due to rounding, use ``np.isclose`` with a tolerance or round your reaction condition arrays to fixed decimals when assigning μ-windows.
* Always save the exact reaction condition grid (a CSV/pickle) you used for plotting so figures are reproducible.

Overlay snapshots
-----------------

If you want to overlay atomic snapshots from the sampled trajectories onto phase diagrams, use the sampled trajectories ``*_mbar.traj`` and map each sampled configuration to its (mu, T) index. Plot small inset structures or highlights using Matplotlib annotations.

