from catemc.geemc import GEEMC
from ase import units
from ase.io import read


MODEL_PATH = '/mnt/nfs/emc_tutorial/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/models/CoNiMgZnMnAlCOH_mace.model'
TARGET_METAL_SPECIES = ['Co', 'Ni', 'Mg', 'Zn', 'Mn', 'Al']


def create_calculators():
    from catemc.repulsive import RepulsivePotential
    from mace.calculators.mace import MACECalculator

    return [
        RepulsivePotential(),
        MACECalculator(model_paths=MODEL_PATH,
                       dispersion=False,
                       device='cuda',
                       default_dtype='float64'),
    ]


def get_fitness_kwargs():
    return dict(
            fmax_list=[0.5, 0.05],
            md_timestep=1 * units.fs,
            md_taut=500 * units.fs,
            md_steps=400,
            md_write_interval=10,
            md_taup=1000 * units.fs,
            compressibility=5e-5 / units.bar,
            neighbor_cutoff=3.5)


def main():
    comp = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5'
    structure = 'bulk'
    system_name = f'{comp}_{structure}'
    dir_name = f'/mnt/nfs/emc_tutorial/{comp}/{structure}'
    input_pop = read(f'{dir_name}/{system_name}_input_pop.traj', index=':')

    emc = GEEMC(
            system_name,
            calculation_dir=dir_name,
            input_pop=input_pop,
            calculator_factory=create_calculators,
            fitness_kwargs=get_fitness_kwargs(),
            fix_indices=None,
            trim_generation=100,
            pressure=1.01325 * units.bar,
            md_generation=10,
            exchange_generation=1,
            bias_starting_step=300,
            crossover_rate=0.1,
            metal_interswap_rate=0.5,
            nonmetal_interswap_rate=0.3,
            intraswap_rate=0.1,
            metal_element_pool=TARGET_METAL_SPECIES,
            nonmetal_element_pool=['O', 'X'],
            raw_score_cutoff=-10000.,
            temperature_key='temperature',
            replica_key='replica',
            acceptance_key='accepted',
            gpu_devices=[0],
            procs_per_device=32,
            scratch_dir=None,
            scratch_sync_interval=None,
            traj_write_interval=50,
            verbose=True)

    emc.run(total_steps=5000)


if __name__ == '__main__':
    main()
