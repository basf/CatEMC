EMC simulations: canonical, Gibbs & grand-canonical
===================================================

This section describes how to run EMC simulations under different ensembles:
* **Canonical (NVT)** — basic MC with fixed composition and cell,
* **Gibbs (GE-EMC)** — for bulk, allowing swaps between replica cells and cell changes (NPT optionally),
* **Grand canonical (GC-EMC)** — for surfaces, sampling adsorbates and vacancies across μ-grid.

Each subsection gives the engine call, required inputs, and a calculation script template.

.. contents::
   :local:
   :depth: 2

Canonical (NVT) usage
---------------------

If you only need to run a canonical (NVT) style evolutionary Monte Carlo with no volume changes and no open-exchange of species between separate cells, you can run a simplified engine (example shown using GEEMC with `pressure=None` or a custom canonical runner).

**Use case**

* Testing mutation/crossover/intraswap logic locally.
* Quick structural exploration without replica exchange.

**Minimal example (canonical-like with GEEMC)**

.. code-block:: python

   from catemc.geemc import GEEMC
   from ase.io import read
   import os

   # read prepared population where temperature is set in atoms.info
   input_pop = read('Co_bulk_input_pop.traj', index=':')
   emc = GEEMC('Co_bulk_test',
               calculation_dir='.',
               calculation_script='calculate_fitness.py',
               input_pop=input_pop,
               pressure=None,           # NVT-like behavior
               trim_generation=100,
               md_generation=0,
               exchange_generation=0,   # no replica-exchange for pure canonical testing
               crossover_rate=0.1,
               intraswap_rate=0.1,
               metal_element_pool=['Co','Ni','Mg','Zn','Mn','Al'],
               nonmetal_element_pool=['O','X'],
               temperature_key='temperature',
               replica_key='replica')
   emc.run(starting_step=1, total_steps=1000)

Gibbs ensemble (GE-EMC) — bulk
------------------------------

GE-EMC coordinates exchanges (insert/remove/mutation) between cells and can run in NPT (use `pressure=...`) or NVT (`pressure=None`) modes. GE-EMC also supports crossover, replica-exchange across assigned temperatures, and optional MD/relaxation.

**Required inputs**

* ``*_input_pop.traj`` with replica metadata.
* ``calculation_script`` for computing fitness per structure.
* Calculator models accessible by the worker nodes.

**Example: run GE-EMC (full example)**

.. code-block:: python

   from catemc.geemc import GEEMC
   from ase.io import read
   import os

   gpu_devices = [0]
   ppg = 10
   os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_devices[0])

   input_pop = read('Co_bulk_input_pop.traj', index=':')
   emc = GEEMC('Co_bulk_run',
               calculation_dir='.',
               calculation_script='calculate_fitness.py',
               input_pop=input_pop,
               fix_indices=None,
               trim_generation=100,
               pressure=0.0,         # NPT-like if nonzero; set None for NVT
               md_generation=10,
               exchange_generation=1,
               bias_starting_step=300,
               crossover_rate=0.1,
               metal_interswap_rate=0.5,
               nonmetal_interswap_rate=0.3,
               intraswap_rate=0.1,
               metal_element_pool=['Co','Ni','Mg','Zn','Mn','Al'],
               nonmetal_element_pool=['O','X'],
               raw_score_cutoff=-10000.,
               temperature_key='temperature',
               replica_key='replica',
               acceptance_key='accepted',
               gpu_devices=gpu_devices,
               procs_per_device=ppg)
   emc.run(starting_step=1, total_steps=5000)

Calculation script template (GE-EMC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Keep the header unchanged — the engine calls the script using these position args:

argv[1] = trajectory path
argv[2] = run_md (0/1)
argv[3] = gpu_device
argv[4] = fix_indices (comma-separated or 'None')

Example template:

.. code-block:: python

   from catemc.calculation_templates import calculate_bulk_fitness
   from catemc.repulsive import RepulsivePotential
   from mace.calculators.mace import MACECalculator
   from ase import units
   from ase.io import read, write
   import sys, os

   #======================Don't change this=====================
   traj_path = sys.argv[1]
   run_md = bool(int(sys.argv[2]))
   gpu_device = sys.argv[3]
   if sys.argv[4] == 'None':
       fix_indices = None
   else:
       fix_indices = [int(i) for i in sys.argv[4].split(',')]
   os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
   dir_name = '/'.join(traj_path.split('/')[:-1])
   atoms = read(traj_path, index=-1)
   atoms.wrap()
   #=========

   calc1 = RepulsivePotential()
   fmax1 = 0.5
   calc2 = MACECalculator(model_paths="/path/to/mace.model", device='cuda', default_dtype='float64')
   fmax2 = 0.05

   calculate_bulk_fitness(
       atoms, traj_path=traj_path,
       calculator_list=[calc1, calc2],
       fmax_list=[fmax1, fmax2],
       target_metal_species=['Co','Ni','Mg','Zn','Mn','Al'],
       run_md=run_md,
       md_timestep=1*units.fs,
       md_steps=50,
       md_write_interval=10,
       pressure=0.,
       fix_indices=fix_indices,
       neighbor_cutoff=3.5,
       raw_score_cutoff=-10000.,
       temperature_key='temperature',
       replica_key='replica')

Grand-canonical (GC-EMC) — surface
----------------------------------

GC-EMC is designed for surface simulations where adsorbates are inserted/removed
and the system is reweighted across a reaction-condition (μ) grid.

**Required inputs**

* ``*_input_pop.traj`` with μ-window metadata.
* ``calculation_script`` for surface fitness (template below).
* ``temperature_dmu_map`` from GE-EMC MBAR (optional but recommended).
* Reaction-condition grid array (``reaction_conditions``).

**Example: launching GC-EMC (abridged)**

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
   E_C = -9.944    # Etot per atom graphite PBE
   E_O2 = -8.768 + 0.101   # Etot incl. ZPE PBE
   E_H2 = -6.773 + 0.267   # Etot incl. ZPE PBE
   dmu_C = np.arange(0., 2.001, 0.1)
   dmu_O = np.arange(-4., -2.999, 0.1)
   dmu_H = np.arange(-1., 0.001, 0.1)
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

Calculation script template (GC-EMC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Header and argv parsing expected (engine passes same args as GE-EMC plus slab indices):

argv[1] = trajectory path
argv[2] = run_md (0/1)
argv[3] = gpu_device
argv[4] = fix_indices
argv[5] = slab_indices

A template is included in the repo and mirrors the bulk template with additional surface-specific calculators and neighbor checks.

Best practices for production runs
----------------------------------

* Use a repulsive potential to stabilize vacancy insertions.
* Keep ``trim_generation`` such that DB backups are frequent enough to avoid losing long runs.
* For GC-EMC prefer multi-try μ-walk (multiple candidate insertions) for robustness.
* If using GPU models (MACE), ensure the cluster or node has the correct drivers/models available and guard script calls to work in CPU-only debugging mode if needed.
