from catemc.gcemc import GCEMC
from acat.adsorption_sites import SlabAdsorptionSites
from ase import data, units
from ase.io import read
from collections import Counter
import math
import numpy as np
import pickle


MODEL_PATH = '/emc/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/models/CoMoFeNiCuNH_mace.model'
TARGET_METAL_SPECIES = ['Co', 'Mo', 'Fe', 'Ni', 'Cu']
TARGET_METAL_COMPOSITION = {'Co': 0.25, 'Mo': 0.45, 'Fe': 0.1,
                            'Ni': 0.1, 'Cu': 0.1}
ALLOWED_ADSORBATE_SPECIES = ['H', 'N', 'NH', 'NH2', 'N2', 'H2', 'NH3']


def create_calculators():
    from mace.calculators.mace import MACECalculator

    return [MACECalculator(model_paths=MODEL_PATH,
                           dispersion=False,
                           device='cuda',
                           default_dtype='float64')]


def get_fitness_kwargs(temperature_dmu_map):
    e_n2 = -16.293 + 0.155
    e_h2 = -6.984 + 0.269
    return dict(
            fmax_list=[0.05],
            target_metal_species=TARGET_METAL_SPECIES,
            target_metal_composition=TARGET_METAL_COMPOSITION,
            allowed_adsorbate_species=ALLOWED_ADSORBATE_SPECIES,
            md_timestep=1 * units.fs,
            md_taut=500 * units.fs,
            md_steps=400,
            md_write_interval=10,
            temperature_dmu_map=temperature_dmu_map,
            element_column_map={'N': 0, 'H': 1},
            element_eref_map={'N': e_n2 / 2, 'H': e_h2 / 2},
            element_y2x_map={'N': 0., 'H': 0.},
            adsorbate_surface_dmax=2.5,
            neighbor_cutoff=2.5,
            raw_score_cutoff=-10000.,
            temperature_key='temperature',
            replica_key='replica')


def main():
    comp = 'Co0.25Mo0.45Fe0.1Ni0.1Cu0.1'
    structure = '111'
    system_name = f'{comp}NH_{structure}'
    dir_name = f'/emc/{comp}/{structure}'
    input_pop = read(f'{dir_name}/{system_name}_input_pop.traj', index=':')
    template = input_pop[0]

    adsorbate_species = ['N', 'H', 'NH', 'NH2', 'N2']
    other_adsorbate_species = ['NH3', 'H2']
    slab_indices = [atom.index for atom in template
                    if atom.symbol in TARGET_METAL_SPECIES]
    fix_indices = [atom.index for atom in template if atom.position[2] < 15.]
    mutable_metal_indices = [atom.index for atom in template
                             if atom.position[2] >= 15.
                             and atom.index in slab_indices]

    volume = np.linalg.norm(np.cross(template.cell[0], template.cell[1])) * 15.
    composition = Counter(template.symbols)
    volume -= sum(
        number * 4 / 3 * math.pi
        * data.covalent_radii[data.atomic_numbers[element]] ** 3
        for element, number in composition.items()
        if element in TARGET_METAL_SPECIES)

    temperatures = sorted({
        atoms.info['key_value_pairs']['temperature'] for atoms in input_pop
    })
    dmu_n = np.arange(-1., 0.001, 0.05).round(2)
    dmu_h = np.arange(-1., 0.001, 0.05).round(2)
    reaction_conditions = np.array(
        np.meshgrid(dmu_n, dmu_h, temperatures)
    ).T.reshape(-1, 3)

    with open(f'/emc/{comp}/bulk/{comp}_bulk_mbar_dmu_estimates.pkl', 'rb') as handle:
        temperature_dmu_map = pickle.load(handle)
    with open(f'{dir_name}/NH_internal_data.pkl', 'rb') as handle:
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
            fitness_kwargs=get_fitness_kwargs(temperature_dmu_map),
            fix_indices=fix_indices,
            trim_generation=100,
            md_generation=10,
            mu_walk_generation=1,
            exchange_generation=1,
            bias_starting_step=300,
            mbar_starting_step=100,
            mbar_sample_stride=10,
            crossover_rate=0.2,
            interswap_rate=0.1,
            slab_intraswap_rate=0.15,
            slab_mutation_rate=0.15,
            adsorbate_insertion_rate=0.1,
            adsorbate_removal_rate=0.1,
            adsorbate_displacement_rate=0.1,
            adsorbate_mutation_rate=0.1,
            adsorbate_surface_dmax=2.5,
            adsorbate_adsorbate_dmin=2.,
            max_adsorbate_species=3,
            max_adsorbate_coverage=1.,
            metal_element_pool=TARGET_METAL_SPECIES,
            nonmetal_element_pool=None,
            bulk_metal_composition=TARGET_METAL_COMPOSITION,
            adsorbate_species=adsorbate_species,
            other_adsorbate_species=other_adsorbate_species,
            mutable_metal_indices=mutable_metal_indices,
            mutable_nonmetal_indices=None,
            slab_indices=slab_indices,
            volume=volume,
            reaction_conditions=reaction_conditions,
            element_column_map={'N': 0, 'H': 1},
            element_eref_map={'N': (-16.293 + 0.155) / 2,
                              'H': (-6.984 + 0.269) / 2},
            element_y2x_map={'N': 0., 'H': 0.},
            temperature_dmu_map=temperature_dmu_map,
            temperature_species_internal_map=temperature_species_internal_map,
            adsorption_sites=adsorption_sites,
            ignore_sites='bridge',
            temperature_key='temperature',
            replica_key='replica',
            acceptance_key='accepted',
            raw_score_cutoff=-10000.,
            gpu_devices=[0],
            procs_per_device=32,
            scratch_dir=None,
            scratch_sync_interval=None,
            traj_write_interval=50,
            verbose=True)

    emc.run(total_steps=2000)


if __name__ == '__main__':
    main()
