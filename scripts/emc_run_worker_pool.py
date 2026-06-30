"""EMC run script with persistent worker pool mode.

Usage:
    python emc_run_worker_pool.py <gpu_devices> <procs_per_gpu> [traj_write_interval]
    python emc_run_worker_pool.py 0 72 50
"""
import sys
import os

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)

from catemc.geemc import EMC
from catemc.worker_pool import WorkerPool
from ase.io import read
from ase import units


def main():
    gpu_devices = [int(di) for di in sys.argv[1].split(",")]
    ppg = int(sys.argv[2])
    traj_write_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_devices[0])

    dopant = "Cu"
    structure = "bulk"
    system_name = "Cu0.1Ru0.9Ox_bulk"
    dir_name = os.path.dirname(os.path.abspath(__file__))
    scratch_dir = "/mnt/scratch/Cu0.1Ru0.9Ox_bulk"
    input_pop_fname = os.path.join(dir_name, f"{system_name}_input_pop.traj")
    target_metal_species = ["Cu", "Ru"]
    target_nonmetal_species = ["O", "X"]
    input_pop = read(input_pop_fname, index=":")

    # --- Worker pool setup ---
    fitness_kwargs = {
        "fmax_list": [0.05],
        "target_metal_species": target_metal_species,
        "md_timestep": 1.0 * units.fs,
        "md_taut": 500.0 * units.fs,
        "md_steps": 200,
        "md_write_interval": 10,
        "max_relax_steps": 100,
        "neighbor_cutoff": 3.5,
        "raw_score_cutoff": -10000.0,
        "temperature_key": "temperature",
        "replica_key": "replica",
    }

    pool = WorkerPool(
        calculator_factory_module="calc_factory_nequip",
        calculator_factory_func="create_calculators",
        calculator_factory_kwargs={},
        fitness_func_module="catemc.calculation_templates",
        fitness_func_name="calculate_bulk_fitness_inprocess",
        fitness_kwargs=fitness_kwargs,
        num_workers=ppg,
        gpu_devices=gpu_devices,
        startup_timeout=600.0,
    )
    print(f"Worker pool ready: {len(input_pop)} replicas, {ppg} workers on GPUs {gpu_devices}")

    # --- EMC setup ---
    emc = EMC(system_name,
              calculation_dir=dir_name,
              input_pop=input_pop,
              fix_indices=None,
              trim_generation=100,
              pressure=0.,
              md_generation=10,
              exchange_generation=1,
              crossover_rate=0.,
              intraswap_rate=1.,
              metal_element_pool=target_metal_species,
              nonmetal_element_pool=target_nonmetal_species,
              raw_score_cutoff=-10000.0,
              temperature_key="temperature",
              replica_key="replica",
              acceptance_key="accepted",
              category_key="dopant_valence",
              gpu_devices=gpu_devices,
              procs_per_device=ppg,
              worker_pool=pool,
              traj_write_interval=traj_write_interval,
              scratch_dir=scratch_dir,
              verbosity="SUMMARY",
              )

    emc.run(total_steps=5000)
    pool.shutdown()


if __name__ == "__main__":
    main()
