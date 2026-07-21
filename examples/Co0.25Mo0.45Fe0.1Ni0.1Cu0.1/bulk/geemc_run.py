from catemc.geemc import GEEMC
from ase import units
from ase.io import read


MODEL_PATH = '/mnt/nfs/emc_tutorial/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/models/CoMoFeNiCuNH_mace.model'
TARGET_METAL_SPECIES = ['Co', 'Mo', 'Fe', 'Ni', 'Cu']


def create_calculators():
    from mace.calculators.mace import MACECalculator

    return [MACECalculator(model_paths=MODEL_PATH,
                           dispersion=False,
                           device='cuda',
                           default_dtype='float64')]


def get_fitness_kwargs():
    return dict(
            fmax_list=[0.05],
            md_timestep=1 * units.fs,
            md_taut=500 * units.fs,
            md_steps=400,
            md_write_interval=10,
            neighbor_cutoff=3.5)


def main():
    comp = 'Co0.25Mo0.45Fe0.1Ni0.1Cu0.1'
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
            pressure=None,
            md_generation=10,
            exchange_generation=1,
            bias_starting_step=300,
            crossover_rate=0.1,
            metal_interswap_rate=0.8,
            intraswap_rate=0.1,
            metal_element_pool=TARGET_METAL_SPECIES,
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
