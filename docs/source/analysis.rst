Analysis
========

This page provides concrete examples for GE-EMC (bulk) and GC-EMC (surface)
postprocessing using the classes in :mod:`catemc.analysis`. The examples mirror
the README tutorial and include recommended parameters and practical tips.

.. contents::
   :local:
   :depth: 2

GE-EMC (bulk) MBAR analysis
---------------------------

Use :class:`catemc.analysis.GEEMCAnalysis` to parse GE-EMC DB backups, extract
Widom insertion samples and potential energies, and compute Δμ estimates
between metal species using MBAR. WLS (weighted least squares) can be used to
enforce antisymmetry and transitivity for the pairwise Δμ estimates.

**Files required / generated**

* Input: GE-EMC DB backup files.
* Output: ``*_mbar_data.pkl``, ``*_mbar_widom_data.pkl``, ``*_mbar_dmu_estimates.pkl``.

**Practical tips**

* Pick ``starting_step`` after burn-in.
* If samples are correlated, increase ``sample_stride``.
* WLS needs enough pairwise entries; if it fails, collect more samples or widen the temperature range.

Example: GE-EMC analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from catemc.analysis import GEEMCAnalysis
   import pickle

   system_name = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5_bulk'
   dir_name = '/mnt/nfs/emc_tutorial/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk'
   ana = GEEMCAnalysis(
           name=system_name,
           calculation_dir=dir_name,
           pop_size=48,
           target_metal_species=['Co','Ni','Mg','Zn','Mn','Al'],
           trim_generation=100,
           temperature_key='temperature',
           replica_key='replica',
           acceptance_key='accepted')

   # Prepare data (reads DB backups; set use_exist=True to use existing pkl)
   ana.prepare_data(use_exist=True, raw_score_cutoff=-10000.)

   # Compute Δμ estimates using MBAR and optional WLS correction
   ana.get_dmu_estimates(starting_step=300, ending_step=1000, sample_stride=10,
                         target_temperatures=[723, 800, 823, 873, 950, 1023, 1100, 1173],
                         wls_correction=True)

   with open(f'{dir_name}/{system_name}_mbar_dmu_estimates.pkl', 'rb') as handle:
       mbar_dmu_estimates = pickle.load(handle)
   for T, d in mbar_dmu_estimates.items():
       print(T)
       print({k: round(v, 2) for k, v in d.items()})

Accessing GE-EMC trajectories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get the trajectory of a specific replica from the GE-EMCAnalysis object:

.. code-block:: python

   # returns an ASE Trajectory object, and writes a file '*_geemc_<replica>.traj'
   ana.get_trajectory(replica=10)

GC-EMC (surface) MBAR analysis
-----------------------------

Use :class:`catemc.analysis.GCEMCAnalysis` to parse GC-EMC DB backups, extract
potential energies and other observables, subsample to reduce correlation,
and perform MBAR reweighting over the μ-grid to obtain structural
expectations.

**Files required / generated**

* Input: GC-EMC DB backups and a ``temperature_dmu_map`` (from bulk GE analysis).
* Output: ``*_mbar_data.pkl``, ``*_mbar_expectations.pkl``, ``*_mbar.traj``, ``*_sampled_indices.npy``.

Practical tips
~~~~~~~~~~~~~~

* Set ``starting_step`` after burn-in and after biasing / μ-walk has equilibrated.
* Increase ``sample_stride`` if correlations are strong.
* Provide vacancy indices (``vacancy_indices``) for vacancy-specific observables.

Example: GC-EMC analysis (abridged)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from catemc.analysis import GCEMCAnalysis
   from ase.io import read
   import numpy as np, pickle

   system_name = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5OH_311'
   dir_name = '/path/to/output'
   ana = GCEMCAnalysis(
           name=system_name,
           calculation_dir=dir_name,
           pop_size=48,
           target_metal_species=['Co','Ni','Mg','Zn','Mn','Al'],
           target_metal_composition={'Co':0.2,'Ni':0.2,'Mg':0.2,'Zn':0.2,'Mn':0.2,'Al':2},
           adsorbate_species=['O','H','OH'],
           trim_generation=100,
           temperature_key='temperature',
           replica_key='replica',
           acceptance_key='accepted')

   ana.prepare_data(use_exist=True, raw_score_cutoff=-10000.)

   E_O2 = -8.768 + 0.101   # Etot incl. ZPE PBE
   E_H2 = -6.773 + 0.267   # Etot incl. ZPE PBE
   dmu_O  = np.arange(-5., -2.999, 0.1)
   dmu_H  = np.arange(-1.0, 0.001, 0.05)
   sim_temperatures = [1023,]
   reaction_conditions = np.array(np.meshgrid(dmu_O, dmu_H, sim_temperatures)).T.reshape(-1, 3)

   with open('/path/to/bulk_wls_dmu_estimates.pkl','rb') as f:
       tdmu_map = pickle.load(f)

   template = read(f'{dir_name}/template_Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O4_311_slab_mace_opt.traj')
   vac_indices = [a.index for a in template if a.symbol in ['O','X']]

   ana.get_structural_expectations(
           z_surf=24.0,
           reaction_conditions=reaction_conditions,
           temperature_dmu_map=tdmu_map,
           element_column_map={'O': 0, 'H': 1},
           element_eref_map={'O': E_O2/2, 'H': E_H2/2},
           element_y2x_map={'O': 4/3, 'H': 0.},
           starting_step=500,
           ending_step=2000,
           sample_stride=50,
           vacancy_indices=vac_indices)

Plotting & postprocessing scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The analysis outputs (pickles, trajectories) are intended to be consumed by the visualization scripts
on the visualize page. Save your ``*_mbar_expectations.pkl`` and the sampled trajectories for plotting.

