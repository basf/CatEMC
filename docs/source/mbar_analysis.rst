MBAR analysis: Δμ estimation & structural ensemble averages
==========================================================

This section explains how to postprocess GE-EMC and GC-EMC outputs using MBAR
to obtain chemical potential differences between metal species (Δμ) and to
compute ensemble structural averages (adsorbate coverages, vacancies, pair counts).

.. contents::
   :local:
   :depth: 2

Why MBAR?
---------

MBAR (Multistate Bennett Acceptance Ratio) lets you combine samples collected
at different temperatures and chemical potential points to compute reweighted
expectations across a target set of conditions. We use MBAR twice:

1. In GE-EMC bulk analysis to estimate pairwise Δμ between metals.
2. In GC-EMC surface analysis to reweight sampled configurations into
   structural expectations across a μ-grid.

GE-EMC: estimate Δμ between metal species
-----------------------------------------

**Inputs**

* DB backups produced during GE-EMC (per-generation ASE DB files).
* Widom insertion samples and potential energies (extracted by the analysis scripts).

**Main steps**

1. Parse DB backups and extract Widom & potential energy samples.
2. Use MBAR to compute free-energy differences mapping to chemical potential differences.
3. Apply WLS (weighted least squares) to correct pairwise Δμ to ensure antisymmetry and transitivity.

**Example: GE-EMCAnalysis pipeline**

.. code-block:: python

   from catemc.analysis import GEEMCAnalysis
   import pickle

   system_name = 'Co_bulk'
   dir_name = './output_bulk'
   ana = GEEMCAnalysis(name=system_name,
                       calculation_dir=dir_name,
                       pop_size=48,
                       target_metal_species=['Co','Ni','Mg','Zn','Mn','Al'],
                       trim_generation=100,
                       temperature_key='temperature',
                       replica_key='replica',
                       acceptance_key='accepted')

   # Parse DB backups (writes *_mbar_data.pkl and *_mbar_widom_data.pkl)
   ana.prepare_data(use_exist=True, raw_score_cutoff=-10000.)

   # Compute dmu estimates across requested temperatures; apply WLS to enforce consistency
   ana.get_dmu_estimates(starting_step=300, ending_step=1000, sample_stride=10,
                         target_temperatures=[723, 800, 823, 873, 950, 1023],
                         wls_correction=True)

   # Load final Δμ mapping
   with open(f'{dir_name}/{system_name}_mbar_dmu_estimates.pkl', 'rb') as handle:
       mbar_dmu_estimates = pickle.load(handle)
   for T, d in mbar_dmu_estimates.items():
       print(T, {k: round(v, 2) for k, v in d.items()})

Notes on WLS correction
^^^^^^^^^^^^^^^^^^^^^^^

WLS reprojects pairwise Δμ onto a consistent chemical potential space so that
d_{i->j} = - d_{j->i} and transitivity (d_{i->k} = d_{i->j} + d_{j->k}) holds
approximately. WLS requires enough pairwise constraints across temperatures to
be robust.

GC-EMC: structural ensemble averages across μ-grid
------------------------------------------------

**Inputs**

* GC-EMC DB backups (biased μ-walks).
* ``temperature_dmu_map`` (from GE-EMC MBAR) mapping each temperature to Δμ values.
* Reaction-condition grid ``reaction_conditions`` used at runtime.

**Main steps**

1. Parse DB backups and extract potential energy and observable samples.
2. Subsample to reduce correlations (``sample_stride``).
3. Use MBAR to reweight samples across the full μ-grid.
4. Compute structural expectations: average adsorbate coverages, pair counts, vacancy concentrations at each grid point.

**Example: GCEMCAnalysis pipeline**

.. code-block:: python

   from catemc.analysis import GCEMCAnalysis
   import numpy as np, pickle
   from ase.io import read

   system_name = 'Co_surface'
   dir_name = './output_surface'
   ana = GCEMCAnalysis(name=system_name,
                       calculation_dir=dir_name,
                       pop_size=48,
                       target_metal_species=['Co','Ni','Mg','Zn','Mn','Al'],
                       adsorbate_species=['O','H','OH'],
                       trim_generation=100,
                       temperature_key='temperature',
                       replica_key='replica',
                       acceptance_key='accepted')

   ana.prepare_data(use_exist=True, raw_score_cutoff=-10000.)

   # Build reaction_conditions grid (example)
   dmu_O  = np.arange(-5., -3.0, 0.1)
   dmu_H  = np.arange(-1.0, 0.0, 0.05)
   sim_temperatures = [1023]
   reaction_conditions = np.array(np.meshgrid(dmu_O, dmu_H, sim_temperatures)).T.reshape(-1, 3)

   # Load temperature_dmu_map from GE-EMC
   with open('./bulk_wls_dmu_estimates.pkl', 'rb') as f:
       tdmu_map = pickle.load(f)

   E_O2 = -8.768 + 0.101   # Etot incl. ZPE PBE
   E_H2 = -6.773 + 0.267   # Etot incl. ZPE PBE
   # Compute structural expectations across the μ-grid
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
   )

Outputs produced
^^^^^^^^^^^^^^^^

* ``*_mbar_data.pkl`` — MBAR raw inputs and processed arrays.
* ``*_mbar_expectations.pkl`` — reweighted expectations per grid point.
* ``*_mbar.traj`` — sampled geometries used for plotting.
* ``*_sampled_indices.npy`` — subsampling indices chosen.

Practical tips & pitfalls
------------------------

* Ensure reaction_conditions used in analysis exactly match the grid used during generation (rounding issues are common).
* If MBAR warns of poor overlap, increase temperature coverage, include more replicas, or collect more samples.
* For large grids, MBAR can be memory/time heavy; subsample aggressively and check convergence.

