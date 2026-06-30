from catemc.calculation_templates import calculate_bulk_fitness
from catemc.repulsive import RepulsivePotential
from mace.calculators.mace import MACECalculator
from ase import units
from ase.io import read
import sys
import os


MODEL_PATH = '/mnt/nfs/emc_tutorial/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/models/CoNiMgZnMnAlCOH_mace.model'
TARGET_METAL_SPECIES = ['Co', 'Ni', 'Mg', 'Zn', 'Mn', 'Al']


def create_calculators():

    calc1 = RepulsivePotential()
    calc2 = MACECalculator(model_paths=MODEL_PATH,
                           dispersion=False,
                           device='cuda',
                           default_dtype='float64')
    return [calc1, calc2]


def get_fitness_kwargs():

    return dict(
            fmax_list=[0.5, 0.05],
            target_metal_species=TARGET_METAL_SPECIES,
            md_timestep=1 * units.fs,
            md_taut=500 * units.fs,
            md_steps=400,
            md_write_interval=10,
            pressure=1.01325 * units.bar,
            md_taup=1000 * units.fs,
            compressibility=5e-5 / units.bar,
            neighbor_cutoff=3.5,
            raw_score_cutoff=-10000.,
            temperature_key='temperature',
            replica_key='replica')


def main():

    traj_path = sys.argv[1]
    run_md = bool(int(sys.argv[2]))
    gpu_device = sys.argv[3]
    fix_indices = None if sys.argv[4] == 'None' else [int(i) for i in sys.argv[4].split(',')]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    atoms = read(traj_path, index=-1)
    atoms.wrap()

    calculate_bulk_fitness(atoms,
                           traj_path=traj_path,
                           calculator_list=create_calculators(),
                           run_md=run_md,
                           fix_indices=fix_indices,
                           **get_fitness_kwargs())


if __name__ == '__main__':
    main()
