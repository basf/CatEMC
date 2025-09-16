from catemc.generate import generate_population
from acat.build.ordering import RandomOrderingGenerator as ROG
from ase.io import read, write
from ase.visualize import view
import numpy as np


comp = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5'
structure = 'bulk'
cwd = f'/mnt/nfs/emc_tutorial/{comp}/{structure}'
target_metal_species = ['Co','Ni','Mg','Zn','Mn','Al']
target_metal_composition = {'Co': 0.2, 'Ni': 0.2, 'Mg': 0.2, 'Zn': 0.2, 'Mn': 0.2, 'Al': 2}
target_nonmetal_species = ['O', 'X']
target_nonmetal_composition = {'O': 3.5, 'X': 0.5}
template = read('Al2CoO4.cif')
cell = template.cell.copy()
for i in range(3):
    cell[i][i] = 8.06
template.set_cell(cell, scale_atoms=True)
template *= (2,2,2)
from acat.build.ordering import RandomOrderingGenerator as ROG
# Distinguish metal and oxygen atoms
metal_ids, oxygen_ids = [], []
for i, atom in enumerate(template):
    if atom.symbol == 'O':
        oxygen_ids.append(i)
    else:
        metal_ids.append(i)
metal_atoms, oxygen_atoms = template[metal_ids], template[oxygen_ids]
# Prepare sorted_ids for sorting atomic indices back to original order
sorted_ids = [0] * len(template)
for k, v in enumerate(metal_ids + oxygen_ids):
    sorted_ids[v] = k
rog = ROG(metal_atoms, elements=target_metal_species,
          composition=target_metal_composition,
          trajectory='template_orderings.traj')
rog.run(num_gen=1)
metal_bulk = read('template_orderings.traj')
bulk = (metal_bulk + oxygen_atoms)[sorted_ids] 

input_pop = generate_population(
    template=bulk,
    pop_size=24,
    target_metal_species=target_metal_species,
    target_metal_composition=target_metal_composition,
    target_nonmetal_species=target_nonmetal_species,
    target_nonmetal_composition=target_nonmetal_composition,
)

rep = 0
for T in [723, 823, 923, 1023]:
    for _ in range(6):
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
