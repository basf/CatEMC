#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Commented starting configurations written by ``catemc init``.

The two bulk templates use the EMT calculator of ASE and a tiny cell, so
that they run in about a minute on a laptop. Replace the ``calculators``
section with your own potential and enlarge the cell, the population and
the number of steps for production runs. The template structure is a file
that the user provides (any format that ASE can read).
"""

_HEADER = '''\
# CatEMC configuration file ({title}).
#
#   catemc check    {{file}}   show what this file resolves to
#   catemc generate {{file}}   build the initial population
#   catemc run      {{file}}   run (or continue) the simulation
#   catemc analyze  {{file}}   post-simulation analysis
#
# Units: K, eV, angstrom, fs, bar. Paths are relative to this file.
# Any entry can be replaced on the command line: -s run.total_steps=200
# Full reference: https://catemc.readthedocs.io/en/latest/config_reference.html

# Prefix of every output file, output folder and statistical ensemble
# (canonical, gibbs or grand_canonical).
name: {name}
calculation_dir: .
ensemble: {ensemble}
'''

_BULK_SYSTEM = '''
# Any composition: one formula, or the two sublattices separately, e.g.
#   composition: Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5X0.5       (X = vacancy)
#   composition: {metals: {Co: 1, Al: 2}, nonmetals: {O: 3.5, X: 0.5}}
composition: Cu0.25Ni0.25Pd0.2Pt0.2Au0.1

# "template" is the structure that is decorated, in any file format that ASE
# can read. This one is a 32-atom fcc cell, written with
#   ase build -x fcc -a 3.8 --cubic -r 2,2,2 Cu template.traj
population:
  template: template.traj
  temperatures: [600, 700, 800, 900]
  replicas_per_temperature: 2
  seed: 1

# One entry per calculator, used one after the other (pre-relaxation first).
# "calculator" is repulsive, emt, mace, mace_mp or any import path such as
# mace.calculators.mace.MACECalculator. The other entries are its arguments.
calculators:
  - calculator: emt
    fmax: 0.1                   # relaxation threshold of this calculator
# - calculator: mace_mp         # MACE-MPA-0, https://github.com/ACEsuit/mace
#   model: medium-mpa-0
#   device: cuda
#   default_dtype: float64
#   fmax: 0.05

# Settings of the MD + relaxation that evaluates every structure.
fitness:
  md_timestep: 2                # fs
  md_taut: 100                  # fs
  md_steps: 20
  neighbor_cutoff: 3.5
  optimizer: LBFGS
'''

_BULK_TAIL = '''
run:
  total_steps: 100

analysis:
  starting_step: 20             # discard the equilibration steps
  sample_stride: 1
  trajectory: {{replicas: [0]}}   # {{name}}_{label}_0.traj
  diagnostics: true             # acceptance ratios per move
  sro: {{cutoff: 3.5}}            # Warren-Cowley short-range order
{dmu}'''

CANONICAL = _HEADER.format(
    title='canonical ensemble', name='CuNiPdPtAu_bulk',
    ensemble='canonical') + _BULK_SYSTEM + '''
# Arguments of catemc.emc.EMC. The move rates are normalised to one.
emc:
  intraswap_rate: 0.9           # swap two unlike atoms within a replica
  crossover_rate: 0.1           # cut-and-splice between two replicas
  md_generation: 10             # MD every N steps
  exchange_generation: 1        # replica exchange every N steps
  trim_generation: 20           # move old rows to a backup database
  procs_per_device: 2           # worker processes (per GPU)
  # gpu_devices: [0]
  # pressure: 1.01325           # bar; NpT instead of NVT
''' + _BULK_TAIL.format(label='emc', dmu='')

GIBBS = _HEADER.format(
    title='Gibbs ensemble', name='CuNiPdPtAu_bulk',
    ensemble='gibbs') + _BULK_SYSTEM + '''
# Arguments of catemc.geemc.GEEMC. The move rates are normalised to one.
emc:
  metal_interswap_rate: 0.8     # swap unlike metals between two replicas
  # nonmetal_interswap_rate: 0.3  # the same on the non-metal sublattice
  intraswap_rate: 0.1           # swap two unlike atoms within a replica
  crossover_rate: 0.1           # cut-and-splice between two replicas
  md_generation: 10             # MD every N steps
  exchange_generation: 1        # replica exchange every N steps
  bias_starting_step: 10        # species-weighted inter-swap from this step on
  trim_generation: 20           # move old rows to a backup database
  procs_per_device: 2           # worker processes (per GPU)
  # gpu_devices: [0]
  # pressure: 1.01325           # bar; NpT instead of NVT
''' + _BULK_TAIL.format(label='geemc', dmu='''\
  dmu:                          # metal chemical-potential differences
    target_temperatures: [600, 650, 700, 800, 900]
''')

GRAND_CANONICAL = _HEADER.format(
    title='grand-canonical ensemble', name='CuNiPd_NH_111',
    ensemble='grand_canonical') + '''
# Composition of the bulk the surface is cut from.
composition: Cu0.5Ni0.25Pd0.25

# Which atoms may move and mutate. Selections: all, none, a list of indices,
# or z_below / z_above / bottom_layers / top_layers / symbols.
regions:
  fix_indices: {bottom_layers: 2}

population:
  # The clean slab that is decorated, in any file format that ASE can read,
  # for example a relaxed slab at the bulk composition.
  template: slab.traj
  temperatures: [573, 673, 773, 873]
  replicas_per_temperature: 6
  seed: 1
  max_adsorbate_species: 1      # one adsorbate type per starting structure
  variants:                     # groups of starting structures
    - {size: 4, adsorbate_species: []}      # clean slabs
    - {adsorbate_species: [N, H, NH]}       # random adlayers (the rest)

calculators:
  - calculator: mace_mp           # MACE-MPA-0, https://github.com/ACEsuit/mace
    model: medium-mpa-0
    dispersion: false
    device: cuda
    default_dtype: float64
    fmax: 0.05

fitness:
  md_timestep: 1                # fs
  md_taut: 500                  # fs
  md_steps: 400
  neighbor_cutoff: 2.5

adsorbates:
  adsorbate_species: [N, H, NH, NH2, N2]  # fragments that are inserted and removed
  other_adsorbate_species: [NH3, H2]      # fragments that can form during MD
  gas_height: 15.0              # height of the accessible volume above the cell area
  internal_partition_functions: # rigid rotor, or a number, or {temperature: value}
    N: {degeneracy: 4}
    H: {degeneracy: 2}
    NH: {degeneracy: 2, symmetry_number: 1}
    NH2: {degeneracy: 2, symmetry_number: 2}
    N2: {degeneracy: 1, symmetry_number: 2}
    NH3: {degeneracy: 1, symmetry_number: 3}
    H2: {degeneracy: 1, symmetry_number: 2}

# Reservoir. "metals": chemical potential of every metal (eV) at every
# temperature of the simulation and the analysis, as printed by
# "catemc analyze" of the bulk Gibbs-ensemble run (analysis.dmu). Only the
# differences between the metals matter. "nonmetals": "reference" is the
# elemental reference energy (half the energy of N2 and H2 here) and the grid
# is relative to it: mu = reference + dmu.
chemical_potentials:
  # metals:
  #   Cu: {573: 0.0, 673: 0.0, 773: 0.0, 800: 0.0, 873: 0.0}
  #   Ni: {573: ..., 673: ..., 773: ..., 800: ..., 873: ...}
  #   Pd: {573: ..., 673: ..., 773: ..., 800: ..., 873: ...}
  nonmetals:
    N: {reference: -8.2425, range: [-1.0, 0.0], step: 0.05}
    H: {reference: -3.253, range: [-1.0, 0.0], step: 0.05}

# Arguments of catemc.gcemc.GCEMC. The move rates are normalised to one.
emc:
  crossover_rate: 0.2
  interswap_rate: 0.1
  slab_intraswap_rate: 0.15
  slab_mutation_rate: 0.15
  adsorbate_insertion_rate: 0.1
  adsorbate_removal_rate: 0.1
  adsorbate_displacement_rate: 0.1
  adsorbate_mutation_rate: 0.1
  max_adsorbate_species: 3
  ignore_sites: bridge
  md_generation: 10
  bias_starting_step: 300       # MBAR bias of the mu-walk from this step on
  mbar_starting_step: 100
  mbar_sample_stride: 10
  gpu_devices: [0]
  procs_per_device: 24

run:
  total_steps: 2000

analysis:
  starting_step: 500
  sample_stride: 50
  z_surf: 22.0                  # atoms above this height are surface atoms
  trajectory: {replicas: [0]}
  diagnostics: true
  # Surface phase diagram. The simulated conditions are always included;
  # reaction_conditions adds more: a grid (mu = reference + values) or single
  # conditions given as absolute chemical potentials (eV), each at T.
  expectations:
    reaction_conditions:
      - N: {reference: -8.2425, range: [-1.0, 0.0], step: 0.025}
        H: {reference: -3.253, range: [-1.0, 0.0], step: 0.025}
        T: 800
      - {N: -9.167, H: -3.881, T: 800}
  # fes:                        # free energies of the sampled structures
  #   bin_width: 0.05
  #   reaction_conditions:
  #     - {N: -9.167, H: -3.881, T: 800}
'''

TEMPLATES = {
    'canonical': CANONICAL.replace('{file}', 'config.yaml'),
    'gibbs': GIBBS.replace('{file}', 'config.yaml'),
    'grand_canonical': GRAND_CANONICAL.replace('{file}', 'config.yaml'),
}
