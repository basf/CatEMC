import os
import sys


def main():

    from catemc.geemc import GEEMC
    from calculate_bulk_fitness import get_fitness_kwargs
    from ase.io import read
    from ase import units

    gpu_devices = [int(di) for di in sys.argv[1].split(',')]
    ppg = int(sys.argv[2])
    traj_write_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    scratch_dir = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != 'None' else None
    scratch_sync_interval = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != 'None' else None
    verbose = bool(int(sys.argv[6])) if len(sys.argv) > 6 else True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_devices[0])

    comp = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5'
    structure = 'bulk'
    system_name = f'{comp}_{structure}'
    dir_name = f'/mnt/nfs/emc_tutorial/{comp}/{structure}'
    input_pop_fname = f'{dir_name}/{system_name}_input_pop.traj'
    target_metal_species = ['Co', 'Ni', 'Mg', 'Zn', 'Mn', 'Al']
    target_nonmetal_species = ['O', 'X']
    input_pop = read(input_pop_fname, index=':')

    emc = GEEMC(system_name,
                calculation_dir=dir_name,
                calculation_script='calculate_bulk_fitness.py',
                input_pop=input_pop,
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
                metal_element_pool=target_metal_species,
                nonmetal_element_pool=target_nonmetal_species,
                raw_score_cutoff=-10000.,
                temperature_key='temperature',
                replica_key='replica',
                acceptance_key='accepted',
                gpu_devices=gpu_devices,
                procs_per_device=ppg,
                use_worker_pool=True,
                fitness_kwargs=get_fitness_kwargs(),
                scratch_dir=scratch_dir,
                scratch_sync_interval=scratch_sync_interval,
                traj_write_interval=traj_write_interval,
                verbose=verbose)

    emc.run(starting_step=1, total_steps=5000)


if __name__ == '__main__':
    main()
