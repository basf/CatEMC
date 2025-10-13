GC-EMC (surface)
=================

This page documents surface GC-EMC structure generation (with μ-window mapping),
running GC-EMC simulations, the calculation script template, and MBAR analysis.
The wording and code examples follow the README tutorial and are ready to run
after adjusting file paths and environment.

.. contents::
   :local:
   :depth: 2

Structure generation (+ μ-window mapping)
----------------------------------------

Generates a set of ``pop_size`` initial on-lattice replicas for surface GC-EMC.
Candidate surface structures are sorted by C/O/H count and then equal numbers
of μ-grid points are assigned to each replica (the μ-window). The μ-window
assignment ensures each replica explores a contiguous portion of the chemical
potential space, and replicas with higher coverage of some species receive
higher μ values in their μ-window.

**Files required and produced**

* Input: a (relaxed) template slab structure file.
* Output: ``*_input_pop.traj`` containing ``pop_size`` on-lattice replicas with
  ``atoms.info`` set (``.info['key_value_pairs']['temperature']``,
  ``.info['data']['mu']`` and ``.info['data']['mu_window']``).

**Practical tips**

* Ensure adsorption sites are correct: use ``SlabAdsorptionSites`` with
  ``surface='autoadsorbate'`` to detect sites on complex surfaces.
* Round μ-points consistently (use ``round_decimals`` in the assign function).
* Validate that assigned μ-windows are contiguous and each replica receives the
  expected number of μ-grid points.

Example: generate a surface population and assign μ-windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from catemc.generate import generate_population, assign_mu_windows
   from acat.adsorption_sites import SlabAdsorptionSites
   from ase.io import read, write
   from ase.visualize import view
   import numpy as np

   comp = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5'
   surface = '311'
   dir_name = f'/mnt/nfs/emc_tutorial/{comp}/{surface}'
   template = read(f'{dir_name}/template_{comp}_{surface}_slab_mace_opt.traj')

   # Choose which atoms are fixed vs mutable
   fix_ids = [a.index for a in template if a.position[2] < 18.8]
   mut_ids = [a.index for a in template if a.index not in fix_ids]

   # Build adsorption site detector
   sas = SlabAdsorptionSites(template, surface='autoadsorbate',
                             composition_effect=False, ignore_sites='bridge')

   target_metal_species = ['Co','Ni','Mg','Zn','Mn','Al']
   target_metal_composition = {'Co':0.2,'Ni':0.2,'Mg':0.2,'Zn':0.2,'Mn':0.2,'Al':2}
   ads_species = ['O','H','OH','CO','CO2','CH','CH2','CH3']

   # Generate candidate on-lattice structures (example pop_size)
   onlat_images = generate_population(
       template=template.copy(),
       pop_size=14,
       allowed_indices=mut_ids,
       target_metal_species=target_metal_species,
       target_metal_composition=target_metal_composition,
       adsorbate_species=ads_species,
       adsorbate_adsorbate_dmin=2.,
       max_adsorbate_species=1,
       adsorption_sites=sas,
   )

   # Assign metadata (replica IDs and temperatures)
   input_images = []
   rep = 0
   for T in [623, 723, 823]:
       for _ in range(14):
           atoms = onlat_images[rep]
           atoms.info = {'key_value_pairs': {}, 'data': {}}
           atoms.info['key_value_pairs']['temperature'] = T
           atoms.info['key_value_pairs']['replica'] = rep
           atoms.info['key_value_pairs']['emc_step'] = 0
           input_images.append(atoms)
           rep += 1

   # Build μ-grid (example for C/O/H)
   E_C = -9.944    # Etot per atom graphite PBE
   E_O2 = -8.768 + 0.101   # Etot incl. ZPE PBE
   E_H2 = -6.773 + 0.267   # Etot incl. ZPE PBE
   dmu_C = np.arange(0., 2.001, 0.1)
   dmu_O  = np.arange(-4., -2.999, 0.1)
   dmu_H  = np.arange(-1., 0.001, 0.05)
   grid = np.array(np.meshgrid(dmu_C, dmu_O, dmu_H)).T.reshape(-1, 3)
   element_column_map = {'C': 0, 'O': 1, 'H': 2}

   # Assign μ-windows to replicas
   assigned_input_pop = assign_mu_windows(
       input_images, reaction_conditions=grid,
       element_column_map=element_column_map, round_decimals=3)

   for img_idx, img in enumerate(assigned_input_pop):
       mu_rep = img.info['data']['mu']
       mu_window = img.info['data']['mu_window']
       T = img.info['key_value_pairs']['temperature']
       print(f'Replica {img_idx} → T = {T}, {len(mu_window)} μ-points in μ-window, currently {tuple(mu_rep)}')

   write(f'{dir_name}/{comp}COH_{surface}_input_pop.traj', assigned_input_pop)
   view(assigned_input_pop)

Run GC-EMC simulation
---------------------

Instantiates ``GCEMC`` and runs a surface GC-EMC simulation. The engine coordinates
intraswap, mutation, interswap, crossover, replica exchange, μ-walk, and optional
MD/relaxation with the calculators specified in the ``calculation_script``.
``reaction_conditions`` defines the μ-grid that the simulation will reweight across.

**Files required**

* ``*_input_pop.traj`` — the initial input population (with μ-window metadata).
* ``calculation_script`` — script to compute fitness per structure (template provided).
* ``temperature_dmu_map`` — Δμ mapping from bulk GE-EMC MBAR.
* ``temperature_internal_map`` — adsorbate internal partition mappings if used.

**Practical tips**

* Provide your own calculation script for production runs: adapt the template to your calculators/models.
* Use a repulsive potential when vacancies exist to avoid bad insertions.
* Prefer multi-try μ-walk (multi-candidate insertion attempts) to single-try for better sampling.

Example: launching GC-EMC (abridged)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from catemc.gcemc import GCEMC
   from ase.io import read
   import numpy as np
   import pickle, sys, os
   from acat.adsorption_sites import SlabAdsorptionSites

   # GPU setup passed via sys.argv in typical cluster usage:
   gpu_devices = [int(di) for di in sys.argv[1].split(',')]
   ppg = int(sys.argv[2])
   os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_devices[0])

   comp = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5'
   surface = '311'
   system_name = f'{comp}COH_{surface}'
   dir_name = f'/mnt/nfs/emc_tutorial/{comp}/{surface}'
   input_pop = read(f'{dir_name}/{system_name}_input_pop.traj', index=':')
   template = read(f'{dir_name}/template_{comp}_{surface}_slab_mace_opt.traj')

   # Build reaction_conditions (mu_C, mu_O, mu_H, T)
   dmu_C = np.arange(0., 2.001, 0.1)
   dmu_O  = np.arange(-4., -2.999, 0.1)
   dmu_H  = np.arange(-1., 0.001, 0.05)
   temperatures = sorted({atoms.info['key_value_pairs']['temperature'] for atoms in input_pop})
   reaction_conditions = np.array(np.meshgrid(dmu_C, dmu_O, dmu_H, temperatures)).T.reshape(-1, 4)

   element_column_map = {'C': 0, 'O': 1, 'H': 2}
   element_eref_map = {'C': E_C, 'O': E_O2/2, 'H': E_H2/2}
   element_y2x_map = {'C': 0., 'O': 4/3, 'H': 0.}

   # Load Δμ map and internal species data
   with open('/path/to/bulk_wls_dmu_estimates.pkl', 'rb') as f:
       tdmu_map = pickle.load(f)
   with open(f'{dir_name}/COH_internal_data.pkl','rb') as handle:
       tsi_map = pickle.load(handle)

   sas = SlabAdsorptionSites(template, surface='autoadsorbate', composition_effect=False, ignore_sites='bridge')

   emc = GCEMC(system_name, calculation_dir=dir_name, calculation_script='calculate_fitness.py',
               input_pop=input_pop, fix_indices=fix_ids, trim_generation=100, md_generation=10,
               mu_walk_generation=1, exchange_generation=1, bias_starting_step=300,
               mbar_starting_step=100, mbar_sample_stride=10, crossover_rate=0.2,
               interswap_rate=100.1, slab_intraswap_rate=0.15, slab_mutation_rate=0.15,
               adsorbate_insertion_rate=0.1, adsorbate_removal_rate=0.1, adsorbate_displacement_rate=0.1,
               adsorbate_mutation_rate=0.1, adsorbate_surface_dmax=2.5, adsorbate_adsorbate_dmin=2.,
               max_adsorbate_species=3, max_adsorbate_coverage=1., metal_element_pool=target_metal_species,
               nonmetal_element_pool=['O','X'], bulk_metal_composition=target_metal_composition,
               adsorbate_species=adsorbate_species, mutable_metal_indices=mut_metal_indices,
               mutable_nonmetal_indices=mut_nonmetal_indices, slab_indices=slab_indices, volume=volume,
               reaction_conditions=reaction_conditions, element_column_map=element_column_map,
               element_eref_map=element_eref_map, element_y2x_map=element_y2x_map,
               temperature_dmu_map=tdmu_map, temperature_species_internal_map=tsi_map,
               adsorption_sites=sas, temperature_key='temperature', replica_key='replica',
               acceptance_key='accepted', raw_score_cutoff=-10000., gpu_devices=gpu_devices,
               procs_per_device=ppg)

   emc.run(starting_step=1, total_steps=5000)

Calculation script template for GC-EMC
-------------------------------------

Below is the calculation script template used by the engine; keep the header (argv parsing and read) intact.

.. code-block:: python

   from catemc.calculation_templates import calculate_surface_fitness
   from catemc.repulsive import RepulsivePotential
   from mace.calculators.mace import MACECalculator
   from ase import units
   from ase.io import read, write
   import numpy as np
   import pickle, sys, os
   from acat.adsorption_sites import SlabAdsorptionSites

   #=====================Don't change this======================
   traj_path = sys.argv[1]
   run_md = bool(int(sys.argv[2]))
   gpu_device = sys.argv[3]
   if sys.argv[4] == 'None':
       fix_indices = None
   else:
       fix_indices = [int(i) for i in sys.argv[4].split(',')]
   if sys.argv[5] == 'None':
       slab_indices = None
   else:
       slab_indices = [int(i) for i in sys.argv[5].split(',')]
   os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
   dir_name = '/'.join(traj_path.split('/')[:-1])
   atoms = read(traj_path, index=-1)
   atoms.wrap()
   #============================================================

   calc1 = RepulsivePotential()
   fmax1 = 0.5
   calc2 = MACECalculator(model_paths="/path/to/mace-mpa.model", dispersion=False, device='cuda', default_dtype='float64')
   fmax2 = 0.25
   calc3 = MACECalculator(model_paths="/path/to/custom_mace.model", dispersion=False, device='cuda', default_dtype='float64')
   fmax3 = 0.05
   E_O2 = -8.768 + 0.101   # Etot incl. ZPE PBE
   E_H2 = -6.773 + 0.267   # Etot incl. ZPE PBE

   calculate_surface_fitness(
       atoms, traj_path=traj_path,
       calculator_list=[calc1, calc2, calc3],
       fmax_list=[fmax1, fmax2, fmax3],
       target_metal_species=['Co','Ni','Mg','Zn','Mn','Al'],
       target_metal_composition={'Co':0.2,'Ni':0.2,'Mg':0.2,'Zn':0.2,'Mn':0.2,'Al':2},
       allowed_adsorbate_species=['O','H','OH','CO','CO2','CH','CH2','CH3'],
       run_md=run_md,
       md_timestep=1*units.fs,
       md_taut=0.5*1000**units.fs,
       md_steps=50,
       md_write_interval=10,
       temperature_dmu_map=tdmu_map,
       element_column_map={'C': 0, 'O': 1, 'H': 2},
       element_eref_map={'C': E_C, 'O': E_O2/2, 'H': E_H2/2},
       element_y2x_map={'C': 0., 'O': 4/3, 'H': 0.},
       adsorption_sites=sas,
       slab_indices=slab_indices,
       fix_indices=fix_indices,
       adsorbate_surface_dmax=2.5,
       neighbor_cutoff=2.5,
       raw_score_cutoff=-10000.,
       temperature_key='temperature',
       replica_key='replica')

GC-EMC MBAR analysis
--------------------

Parses DB backups produced during GC-EMC, extracts potential energy samples, subsamples correlated series and performs MBAR reweighting across the μ-grid. The analysis computes structural expectations (adsorbate coverages, pair counts, vacancy concentration) at each reaction condition and writes files:

* ``*_mbar_data.pkl`` (raw MBAR inputs),
* ``*_mbar_expectations.pkl`` (reweighted structural expectations),
* ``*_mbar.traj`` (sampled geometries), and
* ``*_sampled_indices.npy`` (indices used for sampling).

See the analysis page for example usage of ``GCEMCAnalysis``.

