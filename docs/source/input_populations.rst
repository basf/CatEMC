Input populations
=================

Bulk and surface runs start from ASE trajectories containing one structure per
replica. Every replica must carry at least ``temperature``, ``replica`` and
``emc_step`` in ``atoms.info['key_value_pairs']``. Surface GC-EMC additionally
requires ``atoms.info['data']['mu']`` and ``atoms.info['data']['mu_window']``.

Bulk population
---------------

.. code-block:: python

   from catemc.generate import generate_population
   from ase.io import read, write

   comp = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5'
   target_metal_species = ['Co', 'Ni', 'Mg', 'Zn', 'Mn', 'Al']
   target_metal_composition = {'Co': 0.2, 'Ni': 0.2, 'Mg': 0.2,
                               'Zn': 0.2, 'Mn': 0.2, 'Al': 2}
   target_nonmetal_species = ['O', 'X']
   target_nonmetal_composition = {'O': 3.5, 'X': 0.5}

   template = read('Al2CoO4.cif')
   template *= (2, 2, 2)

   input_pop = generate_population(
       template=template,
       pop_size=48,
       target_metal_species=target_metal_species,
       target_metal_composition=target_metal_composition,
       target_nonmetal_species=target_nonmetal_species,
       target_nonmetal_composition=target_nonmetal_composition,
   )

   rep = 0
   for temperature in [623, 723, 823, 923, 1023, 1123]:
       for _ in range(8):
           input_pop[rep].info = {'key_value_pairs': {}, 'data': {}}
           input_pop[rep].info['key_value_pairs']['temperature'] = temperature
           input_pop[rep].info['key_value_pairs']['replica'] = rep
           input_pop[rep].info['key_value_pairs']['emc_step'] = 0
           rep += 1

   write(f'{comp}_bulk_input_pop.traj', input_pop)

Surface population and μ-windows
--------------------------------

.. code-block:: python

   from catemc.generate import generate_population, assign_mu_windows
   from acat.adsorption_sites import SlabAdsorptionSites
   from ase.io import read, write
   import numpy as np

   comp = 'Rh13_MnO'
   surface = '100'
   template = read('Rh13_MnO_100_sym_0.extxyz')

   target_metal_species = ['Mn', 'Rh']
   target_metal_composition = {'Mn': 0.9326424870466321,
                               'Rh': 0.06735751295336788}
   adsorbate_species = ['O', 'H', 'OH', 'C', 'CO', 'CO2', 'CH', 'CH2', 'CH3']

   z_fix = template.cell[2][2] * 0.5
   mutable_indices = [a.index for a in template if a.position[2] >= z_fix]
   adsorption_sites = SlabAdsorptionSites(template,
                                          surface='autoadsorbate',
                                          composition_effect=False,
                                          ignore_sites='bridge')

   images = generate_population(
       template=template,
       pop_size=32,
       allowed_indices=mutable_indices,
       target_metal_species=target_metal_species,
       target_metal_composition=target_metal_composition,
       target_nonmetal_species=['O', 'X'],
       target_nonmetal_composition={'O': 20, 'X': 1},
       adsorbate_species=adsorbate_species,
       adsorbate_adsorbate_dmin=2.,
       max_adsorbate_species=1,
       adsorption_sites=adsorption_sites,
   )

   rep = 0
   input_pop = []
   for temperature in [623, 723]:
       for _ in range(16):
           atoms = images[rep]
           atoms.info = {'key_value_pairs': {}, 'data': {}}
           atoms.info['key_value_pairs']['temperature'] = temperature
           atoms.info['key_value_pairs']['replica'] = rep
           atoms.info['key_value_pairs']['emc_step'] = 0
           input_pop.append(atoms)
           rep += 1

   dmu_C = np.arange(1., 3.001, 0.1)
   dmu_O = np.arange(-4., -2.999, 0.1)
   dmu_H = np.arange(-1., 0.001, 0.1)
   grid = np.array(np.meshgrid(dmu_C, dmu_O, dmu_H)).T.reshape(-1, 3)

   assigned = assign_mu_windows(input_pop,
                                reaction_conditions=grid,
                                element_column_map={'C': 0, 'O': 1, 'H': 2},
                                round_decimals=3)

   write(f'{comp}COH_{surface}_input_pop.traj', assigned)
