# CatEMC

**CatEMC** is a Python package for running evolutionary Monte Carlo simulations for complex bulk/surface catalyst materials. For surface EMC, it is possible to consider a wide range of reaction conditions in a single EMC run. The repository provides tools to (1) generate populations of structures and μ-windows, (2) run GE-EMC (composition sampling) and GC-EMC (grand-canonical surface sampling) simulations, and (3) analyze simulation data to get chemical potential estimates, structural expectations, and free-energy surfaces.

---

## Quick start — what you need before running anything

1. Python environment: we recommend Python 3.9–3.11 and a virtualenv or conda env. Install key packages used by the code:

   * `autoadsorbate`, `acat`, `ase`, `pymbar`

2. Data layout conventions (expected by many example scripts)

   * `calculation_dir/` — working directory for a run (will contain DB backups, trajectories, pickles).
   * `template_*.traj` — a relaxed template structure for the system (bulk or surface).
   * `*_input_pop.traj` — starting population (replicas) with `.info` pre-populated (temperature, replica id, mu window).
   * precomputed lookup tables (optional): `*_bulk_wls_dmu_estimates.pkl`, `*_internal_data.pkl` used during GC runs.

3. Parallel setup

   * The run scripts assume `CUDA_VISIBLE_DEVICES` is set and that you pass GPU device IDs and `procs_per_device` on the command line. Adapt as needed for CPU-only runs.

---

## Installation (practical)

If you plan to use the code directly from this repo, install in editable mode:

```bash
git clone https://github.com/basf/CatEMC.git
cd CatEMC
pip install -e .
```

---

## High-level workflow (practical view)

A typical GC/GE-EMC campaign follows these steps:

1. **Generate structures** — make a diverse set of starting replicas, choose which atoms are mutable, and assign per-replica μ-windows (the `catemc.generate` utilities do this).
2. **Run EMC** — launch GEEMC for bulk composition sampling or GCEMC for surface grand-canonical sampling. The run will write DB backups every `trim_generation` steps.
3. **Postprocess** — use `catemc.analysis` to (a) extract Widom / trial-swap events and energies and (b) compute MBAR reweighting and WLS-corrected chemical potential differences or structural expectations.
4. **Visualize** — convert MBAR outputs and `*_mbar_expectations.pkl` into FES, heatmaps and lowest-grand-potential structures.

Below I provide concrete, copy-paste-ready scripts you already use, but with a lengthy practical intro before each script explaining what it does, what files it expects, and how to adapt it.

---

## GE-EMC (bulk) — structure generation

**What this script does (practical)**
Creates a set of `pop_size` initial bulk replicas (ASE `Atoms`) that collectively span composition space consistent with your target composition. The generation step organizes atoms so that later GE-EMC swap/mutation moves will operate on well-defined indices and mu-windows can be assigned. The function `generate_population()` will return a list of ASE `Atoms` objects — save these to a `.traj` and inspect visually before running any production GE-EMC. If the composition or lattice parameters are off you will likely fail later during energy evaluations.

**Files created / used**

* Input: a relaxed template CIF or `.traj` (`Al2CoO4.cif` in the example).
* Output: `*_input_pop.traj` containing `pop_size` ASE Atoms with `.info` fields set.

**Practical tips**

* Visualize the created population with `ase.visualize.view` to ensure mutational sites and adsorbates are where you expect.
* If you use a MLIP calculator (MACE), verify one single energy call on the template first.
* Keep `pop_size` a multiple of number of temperatures / replicas you plan to use.

```python
# generation_example.py
# Purpose: generate an initial bulk population for GE-EMC.
from catemc.generate import generate_population
from acat.build.ordering import RandomOrderingGenerator as ROG
from ase.io import read, write
from ase.visualize import view
import numpy as np

comp = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5'
structure = 'bulk'
cwd = f'/mnt/nfs/emc_tutorial/{comp}/{structure}'
target_metal_species = ['Co','Ni','Mg','Zn','Mn','Al']
target_metal_composition = {'Co': 0.2, 'Ni': 0.2, 'Mg': 0.2, 'Zn': 0.2, 'Mn': 0.2, 'Al': 2}
target_nonmetal_species = ['O', 'X']
target_nonmetal_composition = {'O': 3.5, 'X': 0.5}

template = read('Al2CoO4.cif')
# Prepare supercell / ordering
cell = template.cell.copy()
for i in range(3):
    cell[i][i] = 8.06
template.set_cell(cell, scale_atoms=True)
template *= (2,2,2)

# Separate metal and oxygen atoms so we can reorder metals while keeping oxygens fixed
metal_ids, oxygen_ids = [], []
for i, atom in enumerate(template):
    if atom.symbol == 'O':
        oxygen_ids.append(i)
    else:
        metal_ids.append(i)
metal_atoms, oxygen_atoms = template[metal_ids], template[oxygen_ids]
sorted_ids = [0] * len(template)
for k, v in enumerate(metal_ids + oxygen_ids):
    sorted_ids[v] = k

rog = ROG(metal_atoms, elements=target_metal_species,
          composition=target_metal_composition,
          trajectory='template_orderings.traj')
rog.run(num_gen=1)
metal_bulk = read('template_orderings.traj')
bulk = (metal_bulk + oxygen_atoms)[sorted_ids]

input_pop = generate_population(
    template=bulk,
    pop_size=36,
    target_metal_species=target_metal_species,
    target_metal_composition=target_metal_composition,
    target_nonmetal_species=target_nonmetal_species,
    target_nonmetal_composition=target_nonmetal_composition,
)

# assign metadata (temperature/replica/emc step)
rep = 0
for T in [623, 723, 823, 923, 1023, 1123]:
    for _ in range(6):
        input_pop[rep].info = {'key_value_pairs': {}, 'data': {}}
        input_pop[rep].info['key_value_pairs']['temperature'] = T
        input_pop[rep].info['key_value_pairs']['replica'] = rep
        input_pop[rep].info['key_value_pairs']['emc_step'] = 0
        rep += 1

# Save and quick inspect
for img_idx, img in enumerate(input_pop):
    print(f"Replica {img_idx} → T = {img.info['key_value_pairs']['temperature']}")
view(input_pop)
write(f'{cwd}/{comp}_{structure}_input_pop.traj', input_pop)
```

---

## GE-EMC (bulk) — run script

**What this script does (practical)**
This script instantiates `GEEMC` and runs an EM C campaign. It expects `*_input_pop.traj` (the result of the generation step) and a `calculate_fitness.py` (or equivalent) file in the working directory that defines how to evaluate / relax / score structures. The GEEMC engine coordinates mutation, interswap, replica exchange, and optional local MD/relaxation with the calculators you specify inside the `calculate_fitness.py`.

**Files required**

* `*_input_pop.traj` — required
* `calculate_fitness.py` — your scoring function; must accept an ASE `Atoms` and return `raw_score` etc.
* Optional: `MACE model` path if you use an ML calculator.

**Practical tips**

* Test `calculate_fitness.py` on one structure locally (single-threaded) to ensure reproducible energy calls.
* Start with small `total_steps` and smaller `pop_size` to smoke-test resources.
* Use `trim_generation` to control DB checkpoint size; too large → long restarts, too small → many DB files.

```python
# run_geemc.py
from catemc.geemc import GEEMC
from ase.io import read
import numpy as np
import sys, os

# Usage: python run_geemc.py "0" 1
gpu_devices = [int(di) for di in sys.argv[1].split(',')]
ppg = int(sys.argv[2])
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_devices[0])

comp = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5'
system_name = f'{comp}_bulk'
dir_name = f'/mnt/nfs/emc_tutorial/{comp}/bulk'
input_pop_fname = f'{dir_name}/{system_name}_input_pop.traj'
input_pop = read(input_pop_fname, index=':')

emc = GEEMC(system_name,
            calculation_dir=dir_name,
            calculation_script='calculate_fitness.py',
            input_pop=input_pop,
            fix_indices=None,
            trim_generation=100,
            pressure=0.,
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
emc.run(starting_step=14, total_steps=5000)
```

---

## GE-EMC (bulk) — analysis

**What this script does (practical)**
Parses DB backups produced during GEEMC, extracts Widom / trial-swap events, and computes chemical potential differences Δμ between metal species using MBAR. It can also apply a global WLS correction to make all pairwise Δμ consistent.

**Files required**

* Existing DB backups or pickles in `calculation_dir`.
* `*_mbar_dmu_data.pkl` and `*_mbar_energy_data.pkl` will be created if not present.

**Practical tips**

* Set `starting_step` to the step after burn-in when Δμ raw estimates have stabilized (look at the trace plots).
* If MBAR has convergence issues, increase `sample_stride` to reduce correlation or increase total samples by running longer.
* WLS correction needs enough directed Δμ entries to solve the Laplacian robustly.

```python
# analyze_geemc.py
from catemc.analysis import GEEMCAnalysis
import pickle

system_name = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O4_bulk'
dir_name = '/mnt/nfs/emc_tutorial/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O4/bulk'
ana = GEEMCAnalysis(
        name=system_name,
        calculation_dir=dir_name,
        pop_size=48,
        target_metal_species=['Co','Ni','Mg','Zn','Mn','Al'],
        trim_generation=100,
        temperature_key='temperature',
        replica_key='replica',
        acceptance_key='accepted')

# Prepare data (reads DB backups; set use_exist=True to skip parsing)
ana.prepare_data(use_exist=True, raw_score_cutoff=-10000.)
# Compute dmu estimates (MBAR + WLS)
ana.get_dmu_estimates(starting_step=300, ending_step=1000, sample_stride=10,
                      target_temperatures=[723, 800, 823, 873, 950, 1023, 1100, 1173],
                      wls_correction=True)

with open(f'{dir_name}/{system_name}_mbar_dmu_estimates.pkl', 'rb') as handle:
    mbar_dmu_estimates = pickle.load(handle)
for T, d in mbar_dmu_estimates.items():
    print(T)
    print({k: round(v, 2) for k, v in d.items()})
```

---

## GC-EMC (surface) — structure generation (μ-window mapping)

**What this script does (practical)**
Generates an on-lattice starting population for surface GC-EMC. It sorts candidate surface structures by C/O/H count and then assigns equal numbers of μ-grid points to each replica (the μ-window). This step is crucial: it ensures each replica explores a contiguous portion of chemical potential space. If your μ-windowing is poor, MBAR reweighting will have little overlap and analysis will fail.

**Files produced**

* `*_input_pop.traj` with `.info['data']['mu']` and `.info['data']['mu_window']` fields.

**Practical tips**

* For surfaces, ensure `adsorption_sites` are correct (use `SlabAdsorptionSites` with `surface='autoadsorbate'` to choose realistic sites).
* Sort by `C+O+H` or an appropriate composition metric to give each replica relevant μ windows.
* Inspect a few `mu_window` arrays visually during development.

```python
# generate_gc_input.py
from catemc.generate import generate_population
from acat.build import add_adsorbate_to_site
from acat.adsorption_sites import SlabAdsorptionSites
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from ase.visualize import view
import numpy as np
import random, pickle, math, sys, os

comp = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5'
surface = '311'
dir_name = f'/mnt/nfs/emc_tutorial/{comp}/{surface}'
template = read(f'{dir_name}/template_{comp}_{surface}_slab_mace_opt.traj')
fix_ids = [a.index for a in template if a.position[2] < 18.8]
mut_ids = [a.index for a in template if a.index not in fix_ids]
sas = SlabAdsorptionSites(template, surface='autoadsorbate',
                          composition_effect=False,
                          ignore_sites='bridge')

target_metal_species = ['Co','Ni','Mg','Zn','Mn','Al']
target_metal_composition = {'Co':0.2,'Ni':0.2,'Mg':0.2,'Zn':0.2,'Mn':0.2,'Al':2}
ads_species = ['O','H','OH','CO','CO2','CH','CH2','CH3']

# Generate candidate on-lattice structures (3 groups with different O:X counts)
onlat_images = generate_population(
    template=template.copy(),
    pop_size=14,
    allowed_indices=mut_ids,
    target_metal_species=target_metal_species,
    target_metal_composition=target_metal_composition,
    target_nonmetal_species=None,
    target_nonmetal_composition=None,
    adsorbate_species=ads_species,
    adsorbate_adsorbate_dmin=2.,
    max_adsorbate_species=1,
    adsorption_sites=sas,
) + generate_population( template=template.copy(), pop_size=14, ... ) + generate_population(...)

# sort by number of C/O/H atoms and map mu-grid windows (see full example above)
# Save final input_pop with mu windows assigned
write(f'{dir_name}/{comp}COH_{surface}_input_pop.traj', input_pop)
view(input_pop)
```

---

## GC-EMC (surface) — run script

**What this script does (practical)**
Runs surface grand-canonical EMC using `GCEMC`. `reaction_conditions` is your μ-grid (columns: gas μs, T, optional derived elemental μ) and `element_column_map` maps element symbols to their μ column in the grid. `temperature_dmu_map` is the WLS Δμ lookup for bulk metals (used when computing metal-related chemical potentials).

**Files required**

* `*_input_pop.traj` from generation step
* `template_*.traj`
* `*bulk_wls_dmu_estimates.pkl` (from GE-EMC analysis)
* `COH_internal_data.pkl` used for some internal indexing

**Practical tips**

* Run a short GCEMC with `total_steps` small to validate acceptance statistics and mu-walk behavior.
* If μ-walk acceptance is zero for low-replica numbers, inspect `mu_window` centers and temperature assignments.
* Use `mbar_sample_stride` to subsample correlated timeseries for MBAR.

```python
# run_gcemc.py
from catemc.gcemc import GCEMC
from ase.io import read, write
from acat.adsorption_sites import SlabAdsorptionSites
import numpy as np
import pickle, sys, os

gpu_devices = [int(di) for di in sys.argv[1].split(',')]
ppg = int(sys.argv[2])
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_devices[0])

comp = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5'
system_name = f'{comp}COH_311'
dir_name = f'/mnt/nfs/emc_tutorial/{comp}/{surface}'
input_pop = read(f'{dir_name}/{system_name}_input_pop.traj', index=':')
template = read(f'{dir_name}/template_{comp}_{surface}_slab_mace_opt.traj')

# Build reaction_conditions grid (mu_CH4, mu_O, mu_H, T, mu_C)
dmu_CH4 = np.arange(-2.8, -1.299, 0.1)
dmu_O  = np.arange(-3.5, -1.999, 0.1)
dmu_H  = np.arange(-1.0, 0.001, 0.05)
temperatures = sorted({atoms.info['key_value_pairs']['temperature'] for atoms in input_pop})
grid = np.array(np.meshgrid(dmu_CH4, dmu_O, dmu_H, temperatures)).T.reshape(-1, 4)
reaction_conditions = np.hstack([grid, (grid[:,0] + E_CH4 - 4*(grid[:,2] + E_H2/2)).reshape(-1,1)])
reaction_conditions[:, [3, 4]] = reaction_conditions[:, [4, 3]] # Swap columns

element_column_map = {'C': 3, 'O': 1, 'H': 2}
element_eref_map = {'C': 0., 'O': E_O2/2, 'H': E_H2/2}
element_y2x_map = {'C': 0., 'O': 4/3, 'H': 0.}

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

emc.run(starting_step=15, total_steps=5000)
```

---

## GC-EMC (surface) — analysis (structure expectations, FES)

**What this script does (practical)**
Reads GCEMC backups, subsamples correlated series, performs MBAR-based reweighting across your μ-grid, and computes structural expectations (adsorbate coverages, pair bond counts, vacancy concentration) at each reaction condition. It will write `*_mbar_expectations.pkl` and `*_mbar_output.pkl` which you can then use to produce heatmaps and pick representative structures.

**Files required**

* The DB backups and trajectories generated by the GCEMC run
* `bulk_wls_dmu_estimates.pkl` (temperature Δμ maps)
* `reaction_conditions` array saved / reused from the run (same column ordering)

**Practical tips**

* When `sample_stride` is too small you will include strongly correlated samples and MBAR will struggle. Inspect acceptance ratios and effective sample sizes.
* If some reaction condition has zero samples, MBAR reweighting may produce NaNs — either add more simulation at overlapping windows or remove that condition.

```python
# analyze_gcemc.py
from catemc.analysis import GCEMCAnalysis
from ase.io import read, write
import numpy as np, pickle

system_name = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O4OH_311'
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

E_O2 = -8.768 + 0.009
E_H2 = -6.773 + 0.004

dmu_O  = np.arange(-5., -2.999, 0.1)
dmu_H  = np.arange(-1.0, 0.001, 0.05)
temperatures = [1023,]
reaction_conditions = np.array(np.meshgrid(dmu_O, dmu_H, temperatures)).T.reshape(-1, 3)

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
        keep_dims=[0,1],
        target_temperatures=temperatures,
        vacancy_indices=vac_indices)
```
