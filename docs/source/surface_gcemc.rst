Surface GC-EMC
==============

Surface GC-EMC samples surface metal arrangements, lattice vacancies and
adsorbates under a chemical-potential grid. The same production run can be
reweighted to surface phase diagrams and structural expectations.

Accelerated run script
----------------------

.. literalinclude:: ../../scripts/gcemc_run.py
   :language: python

Surface calculation script
--------------------------

.. literalinclude:: ../../scripts/calculate_surface_fitness.py
   :language: python

Run command
-----------

.. code-block:: bash

   python -u gcemc_run.py 0 16 50 /scratch/$USER/catemc_gcemc 100 1

Analysis
--------

.. code-block:: python

   from catemc.analysis import GCEMCAnalysis
   from ase.io import read
   import numpy as np
   import pickle

   system_name = 'Rh13_MnOCOH_100'
   dir_name = '/scicore/home/goedecke/han0011/projects/RhMnO/Rh13_MnO/pet_100'

   analysis = GCEMCAnalysis(
       name=system_name,
       calculation_dir=dir_name,
       pop_size=32,
       target_metal_species=['Mn', 'Rh'],
       target_metal_composition={'Mn': 0.9326424870466321,
                                 'Rh': 0.06735751295336788},
       adsorbate_species=['O', 'H', 'OH', 'C', 'CO', 'CO2', 'CH', 'CH2', 'CH3'],
       trim_generation=100,
       temperature_key='temperature',
       replica_key='replica',
       acceptance_key='accepted')

   analysis.prepare_data(use_exist=True, raw_score_cutoff=-10000.)

   dmu_C = np.arange(1., 3.001, 0.1)
   dmu_O = np.arange(-4., -2.999, 0.1)
   dmu_H = np.arange(-1., 0.001, 0.1)
   temperatures = [723]
   reaction_conditions = np.array(np.meshgrid(dmu_C, dmu_O, dmu_H, temperatures)).T.reshape(-1, 4)

   template = read(f'{dir_name}/Rh13_MnO_100_sym_0.extxyz')
   z_surf = template.cell[2][2] * 0.5
   vacancy_indices = [a.index for a in template
                      if a.symbol in ['O', 'X'] and a.position[2] >= z_surf]

   analysis.get_structural_expectations(
       z_surf=z_surf,
       reaction_conditions=reaction_conditions,
       temperature_dmu_map=None,
       element_column_map={'C': 0, 'O': 1, 'H': 2},
       element_eref_map={'C': -9.944, 'O': (-8.768 + 0.101) / 2, 'H': (-6.773 + 0.267) / 2},
       element_y2x_map={'C': 0., 'O': 1., 'H': 0.},
       starting_step=500,
       ending_step=4000,
       sample_stride=50,
       vacancy_indices=vacancy_indices)

Trajectory extraction
---------------------

.. code-block:: python

   analysis.get_trajectory(replica=10)

.. image:: ../../images/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1NH_111_gcemc_10.gif
   :alt: Surface GC-EMC trajectory
   :align: center
   :width: 420px
