from catemc.generate import assign_mu_windows
from acat.build.ordering import RandomOrderingGenerator as ROG
from acat.build.ordering import SymmetricSlabOrderingGenerator as SSOG
from acat.build.adlayer import special_coverage_pattern as scp
from acat.build.adlayer import max_dist_coverage_pattern as maxdcp
from acat.build import add_adsorbate_to_site
from acat.adsorption_sites import SlabAdsorptionSites
from acat.adsorbate_coverage import SlabAdsorbateCoverage
from acat.settings import site_heights, adsorbate_molecule
from acat.build.adlayer import min_dist_coverage_pattern as mindcp
from ase.constraints import FixAtoms
from ase.geometry import get_layers
from ase.io import read, write
from ase.build import fcc111, fcc111_root
from ase.optimize import BFGS, LBFGS
from ase import units
from ase.visualize import view
from itertools import product
from functools import partial
from collections import Counter
from multiprocessing import Pool
from mace.calculators import mace_mp
from copy import deepcopy
import numpy as np
import random
import pickle
import math
import os


def gen_sqs(atoms, target_concentrations):

    from icet import ClusterSpace
    from icet.tools.structure_generation import (generate_sqs,
                                                 generate_sqs_from_supercells,
                                                 generate_sqs_by_enumeration,
                                                 generate_target_structure)
    from icet.input_output.logging_tools import set_log_config
    set_log_config(level='INFO')

    atoms.pbc = True
    symbols = list(target_concentrations.keys())
    cs = ClusterSpace(atoms, [8.0, 4.0], symbols)
    target_concentrations = target_concentrations 
    sqs_atoms = generate_sqs_from_supercells(cluster_space=cs,
                        supercells=[atoms],
                        n_steps=10000,
                        target_concentrations=target_concentrations,
                        random_start=False)

    return sqs_atoms


if __name__ == '__main__':
    random.seed(1234)
    comp = 'Co0.25Mo0.45Fe0.1Ni0.1Cu0.1'
    surface = '111'
    comp_surf_name = f'{comp}_{surface}'
    dir_name = f'/emc/{comp}/{surface}'
    target_metal_species = ['Co','Mo','Fe','Ni','Cu']
    target_metal_composition = {'Co': 0.25, 'Mo': 0.45, 'Fe': 0.1, 'Ni': 0.1, 'Cu': 0.1}
    ads_species = ['H', 'N', 'NH', 'NH2', 'N2']
    template = fcc111('Cu', a=3.776, size=(6,6,6), vacuum=10., orthogonal=False)
    template.cell[2][2] = 35.
    template.center(axis=2)
    sas = SlabAdsorptionSites(template, surface='autoadsorbate', 
                              composition_effect=False,
                              ignore_sites='bridge')
    slab = template.copy() 
    # Get the indices of the catalyst atoms that are allowed to mutate
    layers = get_layers(slab, (0, 0, 1), tolerance=0.3)[0] # Start from bottom
    fix_ids = np.argwhere(layers<=1).ravel().tolist()
    mut_ids = np.argwhere(layers>1).ravel().tolist()

    if os.path.isfile(f'{dir_name}/sqs_{comp_surf_name}_slab_bot.traj'):
        sqs_slab_bot = read(f'{dir_name}/sqs_{comp_surf_name}_slab_bot.traj')
    else:
        slab_bot = slab[fix_ids]
        rog = ROG(slab_bot, elements=target_metal_species, composition=target_metal_composition, 
                  trajectory=f'{dir_name}/rand_{comp_surf_name}_slab_bot.traj')
        rog.run(num_gen=1)
        rand_slab_bot = read(f'{dir_name}/rand_{comp_surf_name}_slab_bot.traj', index=0)
        rand_slab_bot.cell[2][2] = (11.73-5.55)*2
        rand_slab_bot.center()
        sqs_slab_bot = gen_sqs(rand_slab_bot, target_metal_composition)
        write(f'{dir_name}/sqs_{comp_surf_name}_slab_bot.traj', sqs_slab_bot)
    if os.path.isfile(f'{dir_name}/rand_{comp_surf_name}_slab_top.traj'):
        rand_slab_top = read(f'{dir_name}/rand_{comp_surf_name}_slab_top.traj')
    else:
        slab_top = slab[mut_ids]
        rog = ROG(slab_top, elements=target_metal_species, composition=target_metal_composition, 
                  trajectory=f'{dir_name}/rand_{comp_surf_name}_slab_top.traj')
        rog.run(num_gen=1)
        rand_slab_top = read(f'{dir_name}/rand_{comp_surf_name}_slab_top.traj', index=0)
        write(f'{dir_name}/rand_{comp_surf_name}_slab_top.traj', rand_slab_top)

    slab.symbols[mut_ids] = rand_slab_top.symbols
    slab.symbols[fix_ids] = sqs_slab_bot.symbols
    if os.path.isfile(f'{dir_name}/template_{comp_surf_name}_slab_mace_opt.traj',): 
        slab = read(f'{dir_name}/template_{comp_surf_name}_slab_mace_opt.traj')
    else:
        from mace.calculators.mace import MACECalculator
        calc = MACECalculator(model_paths="/mnt/nfs/emc_tutorial/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/models/CoMoFeNiCuNH_mace.model", 
                              dispersion=False, device='cuda', default_dtype='float64')
        slab.calc = calc
        opt = LBFGS(slab, logfile=None)
        opt.run(fmax=0.01, steps=300)
        write(f'{dir_name}/template_{comp_surf_name}_slab_mace_opt.traj', slab)

    rog = ROG(slab[mut_ids], elements=target_metal_species, 
              composition={m: len([a for a in rand_slab_top if a.symbol == m]) for m in set(rand_slab_top.symbols)},
              trajectory=f'{dir_name}/rand_{comp_surf_name}_slab_top_orderings.traj')
    rog.run(num_gen=56)
    slab_tops = read(f'{dir_name}/rand_{comp_surf_name}_slab_top_orderings.traj', index=':')

    input_images = []
    rep = 0
    for T in [573, 673, 773, 873]:
        cov_sites = [(0.,None), (0.25,'fcc'), (0.5,'fcc'), (0.75,'fcc'), (1.,'fcc'),
                     (0.,None), (0.25,'hcp'), (0.5,'hcp'), (0.75,'hcp'), (1.,'hcp'),
                     (0.33,'rand'), (0.42,'rand'), (0.58,'rand'), (0.67,'rand')]
        cov_sites.sort(key=lambda t: t[0])
        for cov, site in cov_sites:
            atoms = slab.copy()
            if cov in [0.25, 0.5, 0.75, 1.]:
                ads_species = random.choice(['N','H','NH'])
                atoms = scp(atoms, adsorbate_species=ads_species,
                            coverage=cov, surface='fcc111', site_preference=site)
            elif cov > 0.:
                ads_species = random.choice(['N','H','NH'])
                atoms = maxdcp(atoms, adsorbate_species=ads_species, 
                               coverage=cov, surface='fcc111', site_types=['fcc','hcp'])
            atoms.symbols[mut_ids] = slab_tops[rep].symbols
            atoms.info = {'key_value_pairs': {}, 'data': {}}
            atoms.info['key_value_pairs']['temperature'] = T
            atoms.info['key_value_pairs']['replica'] = rep
            atoms.info['key_value_pairs']['emc_step'] = 0
            input_images.append(atoms)
            rep += 1

    dmu_N = np.arange(-1., 0.001, 0.05)
    dmu_H = np.arange(-1., 0.001, 0.05)
    grid = np.array(np.meshgrid(dmu_N, dmu_H)).T.reshape(-1, 2)
    element_column_map = {'N': 0, 'H': 1}
    assigned_input_pop = assign_mu_windows(
            input_images, reaction_conditions=grid,
            element_column_map=element_column_map,
            round_decimals=2)

    for img_idx, img in enumerate(assigned_input_pop):
        mu_rep = img.info['data']['mu']
        mu_window = img.info['data']['mu_window']
        print(f"Replica {img_idx} → T = {img.info['key_value_pairs']['temperature']}," +
              f" {len(mu_window)} μ-points in μ-window, currently {tuple(mu_rep)}")

    view(assigned_input_pop)
    write(f'{dir_name}/{comp}NH_{surface}_input_pop.traj', assigned_input_pop)
