from catemc.geemc import GEEMC
from ase.io import read, write
import numpy as np
import sys
import os


gpu_devices = [int(di) for di in sys.argv[1].split(',')]
ppg = int(sys.argv[2])
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_devices[0])
comp = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5'
structure = 'bulk'
system_name = f'{comp}_{structure}'
dir_name = f'/mnt/nfs/emc_tutorial/{comp}/{structure}'
input_pop_fname = f'{dir_name}/{system_name}_input_pop.traj'
target_metal_species = ['Co','Ni','Mg','Zn','Mn','Al']
target_nonmetal_species = ['O','X']
input_pop = read(input_pop_fname, index=':')

emc = GEEMC(system_name,
            calculation_dir=dir_name,
            calculation_script='calculate_fitness.py',
            input_pop=input_pop,
            fix_indices=None,
            trim_generation=100,
            pressure=None,
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
            procs_per_device=ppg)
emc.run(starting_step=1, total_steps=5000)
