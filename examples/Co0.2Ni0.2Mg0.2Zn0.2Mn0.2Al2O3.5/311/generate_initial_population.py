from catemc.generate import generate_population, assign_mu_windows
from acat.build import add_adsorbate_to_site
from acat.adsorption_sites import SlabAdsorptionSites
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from ase.visualize import view
import numpy as np
import random
import pickle
import math
import sys
import os


comp = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5'
surface = '311'
dir_name = f'/mnt/nfs/emc_tutorial/{comp}/{surface}'
if os.path.isfile(f'{dir_name}/template_{comp}_{surface}_slab_mace_opt.traj',): 
    template = read(f'{dir_name}/template_{comp}_{surface}_slab_mace_opt.traj')
else:
    from mace.calculators.mace import MACECalculator
    template = read(f'{dir_name}/{comp}_{surface}_slab_eqv2_opt.traj')
    template *= (2,1,1)
    template[38].symbol = 'Mn'
    template.constraints = []
    calc = MACECalculator(model_paths="/mnt/nfs/emc_tutorial/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/models/CoNiMgZnMnAlCOH_mace.model", 
                          dispersion=False, device='cuda', default_dtype='float64')
    template.calc = calc
    opt = LBFGS(template, logfile=None)
    opt.run(fmax=0.01, steps=300)
    template = template[[a.index for a in template if not (a.symbol == 'O' and a.position[2] > 28.)]]
    template.wrap()
    write(f'{dir_name}/template_{comp}_{surface}_slab_mace_opt.traj', template)
fix_ids = [a.index for a in template if a.position[2] < 18.8]
mut_ids = [a.index for a in template if a.index not in fix_ids]
sas = SlabAdsorptionSites(template, surface='autoadsorbate', 
                          composition_effect=False,
                          ignore_sites='bridge')
#for st in sas.site_list:
#    add_adsorbate_to_site(template, adsorbate='F', site=st)
#view(template)
#assert False

target_metal_species = ['Co','Ni','Mg','Zn','Mn','Al']
target_metal_composition = {'Co':0.2,'Ni':0.2,'Mg':0.2,'Zn':0.2,'Mn':0.2,'Al':2}
ads_species = ['O','H','OH','CO','CO2','CH','CH2','CH3']
onlat_images = generate_population(
    template=template.copy(),
    pop_size=14,
    allowed_indices=mut_ids,
    target_metal_species=target_metal_species,
    target_metal_composition=target_metal_composition,
    target_nonmetal_species=None,
    target_nonmetal_composition=None,
    adsorbate_species=ads_species,
    adsorbate_adsorbate_dmin=2.,
    max_adsorbate_species=1,
    adsorption_sites=sas,
) + \
generate_population(
    template=template.copy(),
    pop_size=14,
    allowed_indices=mut_ids,
    target_metal_species=target_metal_species,
    target_metal_composition=target_metal_composition,
    target_nonmetal_species=['O','X'],
    target_nonmetal_composition={'O': 20, 'X': 1},
    adsorbate_species=ads_species,
    adsorbate_adsorbate_dmin=2.,
    max_adsorbate_species=1,
    adsorption_sites=sas,
) + \
generate_population(
    template=template.copy(),
    pop_size=14,
    allowed_indices=mut_ids,
    target_metal_species=target_metal_species,
    target_metal_composition=target_metal_composition,
    target_nonmetal_species=['O','X'],
    target_nonmetal_composition={'O': 10, 'X': 1},
    adsorbate_species=ads_species,
    adsorbate_adsorbate_dmin=2.,
    max_adsorbate_species=1,
    adsorption_sites=sas,
)

onlat_images.sort(key=lambda atoms: len([a for a in atoms if a.symbol in ['C','O','H']]))
input_images = []
rep = 0
for T in [623, 723, 823]:
    for _ in range(14):
        atoms = onlat_images[rep]
        atoms.info = {'key_value_pairs': {}, 'data': {}}
        atoms.info['key_value_pairs']['temperature'] = T
        atoms.info['key_value_pairs']['replica'] = rep
        atoms.info['key_value_pairs']['emc_step'] = 0
        input_images.append(atoms)
        rep += 1

E_C = -9.944    # Etot per atom graphite PBE
E_O2 = -8.768 + 0.101   # Etot incl. ZPE PBE
E_H2 = -6.773 + 0.267   # Etot incl. ZPE PBE
dmu_C = np.arange(0., 2.001, 0.1)
dmu_O = np.arange(-4., -2.999, 0.1)
dmu_H = np.arange(-1.0, 0.001, 0.1)
# Mesh: first 3 cols = C, O, H μ
grid = np.array(np.meshgrid(dmu_C, dmu_O, dmu_H)).T.reshape(-1, 3)

element_column_map = {'C': 0, 'O': 1, 'H': 2}
assigned_input_pop = assign_mu_windows(
        input_images, reaction_conditions=grid,
        element_column_map=element_column_map,
        round_decimals=3)

for img_idx, img in enumerate(assigned_input_pop):
    mu_rep = img.info['data']['mu']
    mu_window = img.info['data']['mu_window']
    T = img.info['key_value_pairs']['temperature']
    print(f'Replica {img_idx} → T = {T}, {len(mu_window)} μ-points in μ-window, currently {tuple(mu_rep)}')

view(assigned_input_pop)
write(f'{dir_name}/{comp}COH_{surface}_input_pop.traj', assigned_input_pop)
