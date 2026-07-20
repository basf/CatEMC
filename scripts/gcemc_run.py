from catemc.gcemc import GCEMC
from acat.adsorption_sites import SlabAdsorptionSites
from ase import data, units
from ase.io import read
from collections import Counter
import math
import numpy as np
import pickle


MODEL_PATH = '/scicore/home/goedecke/han0011/model.pt'
TARGET_METAL_SPECIES = ['Mn', 'Rh']
TARGET_METAL_COMPOSITION = {'Mn': 0.9326424870466321,
                            'Rh': 0.06735751295336788}
ALLOWED_ADSORBATE_SPECIES = [
    'O', 'H', 'OH', 'OH2', 'H2', 'O2', 'O2H', 'OOH', 'C', 'CO', 'OC',
    'CO2', 'COH', 'CHO', 'CH', 'CH2', 'CH3', 'CH3O', 'OCH3',
]


def create_calculators():
    from catemc.repulsive import RepulsivePotential
    from metatomic_ase import MetatomicCalculator

    return [
        RepulsivePotential(),
        MetatomicCalculator(MODEL_PATH,
                            device='cuda',
                            variants={'energy': 'react-r2scan'},
                            extensions_directory=None),
    ]


def get_fitness_kwargs():
    E_C = -9.944
    E_O2 = -8.768 + 0.101
    E_H2 = -6.773 + 0.267
    return dict(
            fmax_list=[0.5, 0.05],
            target_metal_species=TARGET_METAL_SPECIES,
            target_metal_composition=TARGET_METAL_COMPOSITION,
            allowed_adsorbate_species=ALLOWED_ADSORBATE_SPECIES,
            md_timestep=1 * units.fs,
            md_taut=500 * units.fs,
            md_steps=50,
            md_write_interval=10,
            temperature_dmu_map=None,
            element_column_map={'C': 0, 'O': 1, 'H': 2},
            element_eref_map={'C': E_C, 'O': E_O2 / 2, 'H': E_H2 / 2},
            element_y2x_map={'C': 0., 'O': 1., 'H': 0.},
            adsorbate_surface_dmax=2.5,
            neighbor_cutoff=2.5,
            update_relaxed_positions=False,
            raw_score_cutoff=-10000.,
            temperature_key='temperature',
            replica_key='replica')


def main():
    comp = 'Rh13_MnO'
    surface = '100'
    system_name = f'{comp}COH_{surface}'
    dir_name = f'/scicore/home/goedecke/han0011/projects/RhMnO/{comp}/fast_pet_{surface}'
    scratch_dir_name = f'/scratch/han0011/catemc_15867815'
    input_pop = read(f'{dir_name}/{comp}COH_{surface}_input_pop.traj', index=':')
    template = read(f'{dir_name}/Rh13_MnO_100_sym_0.extxyz')

    target_nonmetal_species = ['O', 'X']
    adsorbate_species = ['O', 'H', 'OH', 'C', 'CO', 'CO2', 'CH', 'CH2',
                         'CH3', 'CHO', 'COH', 'CH3O']
    other_adsorbate_species = ['OH2', 'H2', 'O2', 'O2H', 'OOH', 'OC', 'OCH3']
    slab_indices = [atom.index for atom in template]
    z_fix = template.cell[2][2] * 0.5
    fix_indices = [atom.index for atom in template if atom.position[2] < z_fix]
    mutable_metal_indices = [
        atom.index for atom in template
        if atom.position[2] >= z_fix and atom.symbol in TARGET_METAL_SPECIES
    ]
    mutable_nonmetal_indices = [
        atom.index for atom in template
        if atom.position[2] >= z_fix and atom.symbol not in TARGET_METAL_SPECIES
    ]

    volume = np.linalg.norm(np.cross(template.cell[0], template.cell[1])) * 16.5
    composition = Counter(template.symbols)
    volume -= sum(
        number * 4 / 3 * math.pi
        * data.covalent_radii[data.atomic_numbers[element]] ** 3
        for element, number in composition.items()
        if element in TARGET_METAL_SPECIES)

    temperatures = sorted({
        atoms.info['key_value_pairs']['temperature'] for atoms in input_pop
    })

    E_C = -9.944
    E_O2 = -8.768 + 0.101
    E_H2 = -6.773 + 0.267
    dmu_C = np.arange(1., 3.001, 0.1)
    dmu_O = np.arange(-4., -2.999, 0.1)
    dmu_H = np.arange(-1., 0.001, 0.1)
    reaction_conditions = np.array(
        np.meshgrid(dmu_C, dmu_O, dmu_H, temperatures)
    ).T.reshape(-1, 4)

    with open(f'{dir_name}/COH_internal_data.pkl', 'rb') as handle:
        temperature_species_internal_map = pickle.load(handle)

    adsorption_sites = SlabAdsorptionSites(
            template,
            surface='autoadsorbate',
            composition_effect=False,
            ignore_sites='bridge')

    emc = GCEMC(
            system_name,
            calculation_dir=dir_name,
            input_pop=input_pop,
            calculator_factory=create_calculators,
            fitness_kwargs=get_fitness_kwargs(),
            fix_indices=fix_indices,
            trim_generation=100,
            md_generation=10,
            mu_walk_generation=1,
            exchange_generation=1,
            bias_starting_step=300,
            mbar_starting_step=100,
            mbar_sample_stride=10,
            crossover_rate=0.1,
            interswap_rate=0.,
            slab_intraswap_rate=0.4,
            slab_mutation_rate=0.,
            adsorbate_insertion_rate=0.125,
            adsorbate_removal_rate=0.125,
            adsorbate_displacement_rate=0.125,
            adsorbate_mutation_rate=0.125,
            adsorbate_surface_dmax=2.5,
            adsorbate_adsorbate_dmin=2.,
            max_adsorbate_species=3,
            max_adsorbate_coverage=1.,
            metal_element_pool=TARGET_METAL_SPECIES,
            nonmetal_element_pool=target_nonmetal_species,
            bulk_metal_composition=TARGET_METAL_COMPOSITION,
            adsorbate_species=adsorbate_species,
            other_adsorbate_species=other_adsorbate_species,
            mutable_metal_indices=mutable_metal_indices,
            mutable_nonmetal_indices=mutable_nonmetal_indices,
            slab_indices=slab_indices,
            volume=volume,
            reaction_conditions=reaction_conditions,
            element_column_map={'C': 0, 'O': 1, 'H': 2},
            element_eref_map={'C': E_C, 'O': E_O2 / 2, 'H': E_H2 / 2},
            element_y2x_map={'C': 0., 'O': 1., 'H': 0.},
            temperature_dmu_map=None,
            temperature_species_internal_map=temperature_species_internal_map,
            adsorption_sites=adsorption_sites,
            ignore_sites='bridge',
            temperature_key='temperature',
            replica_key='replica',
            acceptance_key='accepted',
            raw_score_cutoff=-10000.,
            gpu_devices=[0],
            procs_per_device=16,
            worker_startup_timeout=600.,
            worker_result_timeout=3600.,
            scratch_dir=scratch_dir_name,
            scratch_sync_interval=100,
            traj_write_interval=50,
            verbose=True)

    emc.run(total_steps=5000)


if __name__ == '__main__':
    main()
