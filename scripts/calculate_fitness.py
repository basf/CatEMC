from catemc.calculation_templates import calculate_surface_fitness
from catemc.repulsive import RepulsivePotential
from metatomic_ase import MetatomicCalculator
from ase import units
from ase.io import read
import pickle
import sys
import os


MODEL_PATH = '/scicore/home/goedecke/han0011/model.pt'

E_C = -9.944
E_O2 = -8.768 + 0.101
E_H2 = -6.773 + 0.267

TARGET_METAL_SPECIES = ['Mn', 'Rh']
TARGET_METAL_COMPOSITION = {'Mn': 0.9326424870466321, 'Rh': 0.06735751295336788}
ALLOWED_ADSORBATE_SPECIES = ['O', 'H', 'OH', 'OH2', 'H2', 'O2', 'O2H', 'OOH',
                             'C', 'CO', 'OC', 'CO2', 'COH', 'CHO', 'CH', 'CH2',
                             'CH3', 'CH3O', 'OCH3']


def create_calculators():

    calc1 = RepulsivePotential()
    calc2 = MetatomicCalculator(MODEL_PATH, device='cuda',
                                variants={'energy': 'react-r2scan'},
                                extensions_directory=None)
    return [calc1, calc2]


def get_fitness_kwargs():

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

    traj_path = sys.argv[1]
    run_md = bool(int(sys.argv[2]))
    gpu_device = sys.argv[3]
    fix_indices = None if sys.argv[4] == 'None' else [int(i) for i in sys.argv[4].split(',')]
    slab_indices = None if sys.argv[5] == 'None' else [int(i) for i in sys.argv[5].split(',')]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    dir_name = '/'.join(traj_path.split('/')[:-1])
    atoms = read(traj_path, index=-1)
    atoms.wrap()

    with open(f'{dir_name}/Rh13_MnO_100_sites.pkl', 'rb') as handle:
        sas = pickle.load(handle)

    kwargs = get_fitness_kwargs()
    kwargs['adsorption_sites'] = sas
    kwargs['slab_indices'] = slab_indices
    calculate_surface_fitness(atoms,
                              traj_path=traj_path,
                              calculator_list=create_calculators(),
                              run_md=run_md,
                              fix_indices=fix_indices,
                              **kwargs)


if __name__ == '__main__':
    main()
