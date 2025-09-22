from catemc.generate import generate_population
from acat.build.ordering import RandomOrderingGenerator as ROG
from ase.io import read, write
from ase.visualize import view
import numpy as np


comp = 'Co0.25Mo0.45Fe0.1Ni0.1Cu0.1'
structure = 'bulk'
cwd = f'/mnt/nfs/emc_tutorial/{comp}/{structure}'
target_metal_species = ['Co','Mo','Fe','Ni','Cu']
target_metal_composition = {'Co': 0.25, 'Mo': 0.45, 'Fe': 0.1, 'Ni': 0.1, 'Cu': 0.1}
template = bulk('Cu', 'fcc', a=3.776)
template *= (8,8,8)

input_pop = generate_population(
    template=template,
    pop_size=48,
    target_metal_species=target_metal_species,
    target_metal_composition=target_metal_composition,
)

rep = 0
for T in [573, 650, 723, 800, 873, 950, 1023, 1100, 1173, 1250, 1323, 1400]:
    for _ in range(4):
        input_pop[rep].info = {'key_value_pairs': {}, 'data': {}}
        input_pop[rep].info['key_value_pairs']['temperature'] = T
        input_pop[rep].info['key_value_pairs']['replica'] = rep
        input_pop[rep].info['key_value_pairs']['emc_step'] = 0
        rep += 1
# Report
for img_idx, img in enumerate(input_pop):
    print(f"Replica {img_idx} → T = {img.info['key_value_pairs']['temperature']}")
view(input_pop)
write(f'{cwd}/{comp}_{structure}_input_pop.traj', input_pop)
