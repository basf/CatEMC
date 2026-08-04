Bulk GE-EMC
===========

Bulk GE-EMC estimates relative metal chemical-potential differences for a target
bulk composition. These values can be used as the metal reservoir in surface
GC-EMC.

Accelerated run script
----------------------

.. literalinclude:: ../../scripts/geemc_run.py
   :language: python

Bulk calculation script
-----------------------

.. literalinclude:: ../../scripts/calculate_bulk_fitness.py
   :language: python

Run command
-----------

.. code-block:: bash

   python -u geemc_run.py 0 16 50 /scratch/$USER/catemc_geemc 100 1

Analysis
--------

.. code-block:: python

   from catemc.analysis import GEEMCAnalysis
   import pickle

   system_name = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5_bulk'
   dir_name = '/mnt/nfs/emc_tutorial/Co0.2Ni0.2Mg0.2Zn0.2Al2O3.5/bulk'

   analysis = GEEMCAnalysis(
       name=system_name,
       calculation_dir=dir_name,
       pop_size=48,
       target_metal_species=['Co', 'Ni', 'Mg', 'Zn', 'Mn', 'Al'],
       trim_generation=100,
       temperature_key='temperature',
       replica_key='replica',
       acceptance_key='accepted')

   analysis.prepare_data(use_exist=True, raw_score_cutoff=-10000.)
   analysis.get_dmu_estimates(starting_step=300,
                              ending_step=1000,
                              sample_stride=10,
                              target_temperatures=[723, 800, 823, 873, 950, 1023],
                              wls_correction=True)

   with open(f'{dir_name}/{system_name}_mbar_dmu_estimates.pkl', 'rb') as handle:
       dmu_estimates = pickle.load(handle)

   for temperature, values in dmu_estimates.items():
       print(temperature, values)

Trajectory extraction
---------------------

.. code-block:: python

   analysis.get_trajectory(replica=10)

.. image:: ../../images/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O4_bulk_geemc_10.gif
   :alt: Bulk GE-EMC trajectory
   :align: center
   :width: 420px
