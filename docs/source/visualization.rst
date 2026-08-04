Visualization
=============

The MBAR expectation files are ordinary pickle dictionaries and can be plotted
with Matplotlib. The examples below follow the phase-diagram style used for
CatEMC surface analysis.

Surface metal enrichment
------------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   import pickle
   from matplotlib.colors import TwoSlopeNorm

   system_name = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5OH_311'
   with open(f'{system_name}_mbar_expectations.pkl', 'rb') as handle:
       expectations = pickle.load(handle)

   metal_species = ['Co', 'Ni', 'Mg', 'Zn', 'Mn', 'Al']
   metal_composition = {'Co': 0.2 / 3, 'Ni': 0.2 / 3, 'Mg': 0.2 / 3,
                        'Zn': 0.2 / 3, 'Mn': 0.2 / 3, 'Al': 2 / 3}

   T = 1023
   dmu_O = np.arange(-5., -2.999, 0.02)
   dmu_H = np.arange(-1., 0.001, 0.01)
   grid = np.array(np.meshgrid(dmu_O, dmu_H, [T])).T.reshape(-1, 3)

   fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.5),
                            gridspec_kw={'wspace': 0, 'hspace': 0})

   norm = TwoSlopeNorm(vmin=np.log10(0.2), vcenter=0., vmax=np.log10(5.0))
   for ax, metal in zip(axes.ravel(), metal_species):
       values = expectations[f'x_surf_{metal}'] / metal_composition[metal]
       Z = np.empty((len(dmu_H), len(dmu_O)))
       for iy, muH in enumerate(dmu_H):
           for ix, muO in enumerate(dmu_O):
               idx = np.where(np.isclose(grid[:, 0], muO) & np.isclose(grid[:, 1], muH))[0]
               Z[iy, ix] = values[idx[0]]
       cf = ax.contourf(dmu_O, dmu_H, np.log10(Z), levels=200, norm=norm)
       ax.set_title(metal)

   fig.colorbar(cf, ax=axes.ravel().tolist(), label=r'$c_m^{surf}/c_m^{bulk}$')
   plt.show()

.. image:: ../../images/metal_spd.png
   :alt: Surface metal phase diagram
   :align: center
   :width: 620px

Adsorbate coverage
------------------

.. image:: ../../images/adsorbate_spd.png
   :alt: Adsorbate phase diagram
   :align: center
   :width: 760px
