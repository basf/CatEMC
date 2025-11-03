#!/usr/bin/env python
# -*- coding: utf-8 -*-
from acat.build.ordering import RandomOrderingGenerator as ROG
from acat.build.ordering import SymmetricSlabOrderingGenerator as SSOG
from acat.adsorption_sites import SlabAdsorptionSites
from acat.build.adlayer import min_dist_coverage_pattern as mindcp
from acat.build.adlayer import max_dist_coverage_pattern as maxdcp
from ase.io import read, write
from ase.formula import Formula
from itertools import combinations
from collections import Counter
import numpy as np
import warnings
import random
import pickle
import sys
import os


def generate_population(
        template,
        pop_size,
        input_type='slab',
        target_metal_species=None,
        target_metal_composition=None,
        target_nonmetal_species=None,
        target_nonmetal_composition=None,
        allowed_indices=None,
        valences=None,
        adsorbate_species=None,
        adsorbate_coverage=None,
        adsorbate_adsorbate_dmin=None,
        max_adsorbate_species=None,
        adsorption_sites=None,
        adsorption_dict={},
        symmetry_constraint=False,
        symmetry_dict={},
        trajectory=None,
        append_trajectory=False,
        seed=None):
    """
    Generate a population of random bulk/slab configurations.

    inputs:
        structures (list): a list of structures (ase.Atoms or pymatgen.core.Structure) 
                         as the initial GA population. Generate random structures if
                         the structures are not provided (or complete the population 
                         if only part of structures are provided).
        pop_size (int): GA population size
        target_metal_species (list or str): a list of possible metal species to present
                         in the catalyst.
        target_metal_composition (dict): a dictionary that maps each metal species to
                         the corresponding concentration.
        target_nonmetal_species (list or str): a list of possible non-metal species to 
                         present in the catalyst.
        target_nonmetal_composition (dict): a dictionary that maps each non-metal 
                         species to the corresponding concentrations.
        allowed_indices (list): a list of atom indices that are allowed to mutate.
        valences (dict): a dictionary that maps the each species to the corresponding 
                         charge state to ensure charge neutrality of the structures.
        adsorbate_species (list or str): a list of possible adsorbate species to adsorb
                         on the catalyst surface.
        adsorbate_coverage (float): fix to this adsorbate coverage.
        adsorbate_adsorbate_dmin (float): the radius of the sphere inside which no other 
                         adsorbates should be found. 
        max_adsorbate_species (int): the maximum allowed types of adsorbate species 
                         (excluding vacancies) for a single structure. Allow all 
                         adsorbatae species if not specified.
        adsorption_sites (acat.adsorption_sites.Cluster/SlabAdsorptionSites object or 
                         the corresponding pickle file path, default None): provide the 
                         ACAT built-in adsorption sites class to accelerate the adlayer 
                         pattern generation.
        adsorption_dict (dict): parameters for identifying adsorption sites.
        symmetry_constraint (bool): whether to apply symmetry constraint. Currently only
                         support generating symmetric clusters.
        symmetry_dict (bool): paramters for generating symmetric clusters.
        trajectory (str): path for saving the structures in the inital population.
        seed (int): random seed for reproducibility. 
    """
    if seed is None:
        seed = random.randrange(sys.maxsize)
    random.seed(seed)
    images = []
    ngen = pop_size

    if target_metal_species is None:
        target_metal_species = []
    elif isinstance(target_metal_species, str):
        target_metal_species = [target_metal_species]
    if (target_nonmetal_species is not None) and isinstance(target_nonmetal_species, str):
        target_nonmetal_species = [target_nonmetal_species]
    if ngen <= 0:
        gen_images = []
    elif len(target_metal_species) > 1:
        # Distinguish metal and non-metal atoms
        metal_ids, nonmetal_ids = [], []
        for i, atom in enumerate(template):
            if atom.symbol in target_metal_species:
                if (allowed_indices is None) or (i in allowed_indices):
                    metal_ids.append(i)
                else:
                    nonmetal_ids.append(i)
            else:
                nonmetal_ids.append(i)
        metal_atoms = template[metal_ids]
        if nonmetal_ids:
            nonmetal_atoms = template[nonmetal_ids]
            # Prepare sorted_ids for sorting atomic indices back to original order
            sorted_ids = [0] * len(template)
            for k, v in enumerate(metal_ids + nonmetal_ids):
                sorted_ids[v] = k
        if not symmetry_constraint:
            from acat.build.ordering import RandomOrderingGenerator as ROG
            gen = ROG(metal_atoms, elements=target_metal_species,
                      composition=target_metal_composition,
                      trajectory='metal_orderings.traj')
            gen.run(num_gen=ngen)
        else:
            if input_type in ['cluster']:
                from acat.build.ordering import SymmetricClusterOrderingGenerator as SCOG
                symmetry_dict['atoms'] = metal_atoms
                symmetry_dict['elements'] = target_metal_species
                symmetry_dict['composition'] = target_metal_composition
                symmetry_dict['trajectory'] = 'metal_orderings.traj'
                symmetry_dict['save_groups'] = True
                gen = SCOG(**symmetry_dict)
                gen.run(max_gen=ngen)
            elif input_type in ['slab', 'slab+ads', 'surface']:
                from acat.build.ordering import SymmetricSlabOrderingGenerator as SSOG
                symmetry_dict['atoms'] = metal_atoms
                symmetry_dict['elements'] = target_metal_species
                symmetry_dict['composition'] = target_metal_composition
                symmetry_dict['trajectory'] = 'metal_orderings.traj'
                symmetry_dict['save_groups'] = True
                gen = SSOG(**symmetry_dict)
                gen.run(max_gen=ngen)
            else:
                raise NotImplementedError
        gen_images = read('metal_orderings.traj', index=':')
        if nonmetal_ids:
            gen_images = [(a + nonmetal_atoms)[sorted_ids] for a in gen_images]
        os.remove('metal_orderings.traj')
        if (target_metal_composition is None) and (valences is not None):
            max_anion_valence = len(nonmetal_ids) * max(abs(v) for v in valences.values() if v < 0)
            for atoms in gen_images:
                ctr = Counter(atoms.symbols)
                cation_valence = sum([ctr[k]*v for k, v in valences.items() if v > 0])
                sorted_tups = sorted([(k, v) for k, v in valences.items() if v > 0], key=lambda x: x[1])
                while cation_valence > max_anion_valence:
                    sorted_ids = sorted(random.sample(range(len(sorted_tups)), 2))
                    metal_pair = [sorted_tups[i][0] for i in sorted_ids]
                    metal1_ids = [a.index for a in atoms if a.symbol == metal_pair[1]]
                    if allowed_indices is not None:
                        metal1_ids = [i for i in metal1_ids if i in allowed_indices]
                    if not metal1_ids:
                        continue
                    mut_id = random.choice(metal1_ids)
                    atoms[mut_id].symbol = metal_pair[0]
                    ctr[metal_pair[0]] += 1
                    ctr[metal_pair[1]] -= 1
                    cation_valence = sum([ctr[k]*v for k, v in valences.items() if v > 0])
    else:
        gen_images = [template for _ in range(ngen)]

    if target_nonmetal_species is not None:
        # Distinguish metal and non-metal atoms                                                
        nonmetal_ids, metal_ids = [], []
        for i, atom in enumerate(template):
            if atom.symbol in target_nonmetal_species:
                if (allowed_indices is None) or (i in allowed_indices):
                    nonmetal_ids.append(i)
                else:
                    metal_ids.append(i)
            else:
                metal_ids.append(i)
        nonmetal_atoms = template[nonmetal_ids]
        if metal_ids:
            metal_images = [atoms[metal_ids] for atoms in gen_images]
            # Prepare sorted_ids for sorting atomic indices back to original order
            sorted_ids = [0] * len(template)
            for k, v in enumerate(nonmetal_ids + metal_ids):
                sorted_ids[v] = k
        from acat.build.ordering import RandomOrderingGenerator as ROG
        gen = ROG(nonmetal_atoms, elements=target_nonmetal_species,
                  composition=target_nonmetal_composition,
                  trajectory='nonmetal_orderings.traj')
        gen.run(num_gen=ngen)
        nonmetal_images = read('nonmetal_orderings.traj', index=':')
        if metal_ids:
            gen_images = [(a + metal_images[i])[sorted_ids] for
                          i, a in enumerate(nonmetal_images)]
        else:
            gen_images = nonmetal_images
        os.remove('nonmetal_orderings.traj')
        if valences is not None:
            assert len([v for v in valences.values() if v < 0]) == 1
            for atoms in gen_images:
                ctr = Counter(atoms.symbols)
                cation_valence = sum([ctr[k]*v for k, v in valences.items() if v > 0])
                anion = [k for k, v in valences.items() if v < 0][0]
                anion_valence = abs(ctr[anion] * valences[anion])
                nadd = int((cation_valence - anion_valence) / abs(valences[anion]))
                if nadd > 0:
                    options = [a.index for a in atoms if a.symbol == 'X']
                    if allowed_indices is not None:
                        options = [i for i in options if i in allowed_indices]
                    add_ids = np.random.choice(options, nadd, replace=False)
                    atoms.symbols[add_ids] = anion
                elif nadd < 0:
                    options = [a.index for a in atoms if a.symbol == anion]
                    if allowed_indices is not None:
                        options = [i for i in options if i in allowed_indices]
                    del_ids = np.random.choice(options, -nadd, replace=False)
                    atoms.symbols[del_ids] = 'X'

    if adsorbate_species is not None:
        if adsorption_sites is None:
            if True in template.pbc:
                sas = SlabAdsorptionSites(template, **adsorption_dict)
            else:
                sas = ClusterAdsorptionSites(template, **adsorption_dict)
        else:
            if isinstance(adsorption_sites, str):
                import pickle
                with open(adsorption_sites, 'rb') as f:
                    sas = pickle.load(f)
            else:
                sas = adsorption_sites

        if adsorbate_adsorbate_dmin is None:
            adsorbate_adsorbate_dmin = 2.

        tmp_images = []
        for atoms in gen_images:
            if max_adsorbate_species is None:
                ads_species = adsorbate_species
            else:
                ads_species = list(np.asarray(adsorbate_species)[np.random.choice(
                        len(adsorbate_species), size=max_adsorbate_species, replace=False)])
            if adsorbate_coverage is None:
                dmin = random.uniform(adsorbate_adsorbate_dmin, 5.) 
                pattern = mindcp(atoms, ads_species,
                                 min_adsorbate_distance=dmin,
                                 adsorption_sites=sas)
            else:
                pattern = maxdcp(atoms, ads_species,
                                 coverage=adsorbate_coverage,
                                 adsorption_sites=sas)
            tmp_images.append(pattern)
        gen_images = tmp_images

    input_pop = []
    for atoms in images + gen_images:
        if not atoms.info.get('seed'):
            atoms.info['seed'] = seed
        input_pop.append(atoms)

    if trajectory is not None:
        if append_trajectory:
            input_pop = read(trajectory, index=':') + input_pop
        write(trajectory, input_pop)

    return input_pop


def assign_mu_windows(input_pop, reaction_conditions, element_column_map, round_decimals=3):
    """
    Assign μ-windows to a list of ase.Atoms objects based on their non-metal counts.
    Modify each Atoms.info['data'] to include:
        - 'mu'        : numpy array (center mu for that replica)
        - 'mu_window' : numpy array (rows assigned to that replica)
    The μ-windows are divided evenly from the whole chemical potential space.

    Parameters
    ----------
    input_pop: list of ase.Atoms objects
        list of ase.Atoms objects (replicas).

    reaction_conditions: numpy.ndarray
        2D numpy array, shape (n_conditions, n_elements). Each row is a μ-point.

    element_column_map: dict of (str, int) pairs
        Dict mapping non-metal element names to column indices in reaction_conditions,

    round_decimals: int, default 3
        Number of decimals to round the chosen center μ stored in atoms.info['data']['mu'].

    """

    atoms_list = list(input_pop)
    n_images = len(atoms_list)
    if n_images == 0:
        return atoms_list

    reaction_conditions = np.asarray(reaction_conditions)
    if reaction_conditions.ndim != 2:
        raise ValueError('reaction_conditions must be a 2D array (n_points, n_cols)')
    n_grid = reaction_conditions.shape[0]

    # 1) Count nonmetal atoms in each structure using element_column_map keys as species list
    nonmetal_species = set(element_column_map.keys())
    counts = []
    for i, atoms in enumerate(atoms_list):
        cnt = sum(1 for a in atoms if a.symbol in nonmetal_species)
        counts.append((i, cnt))

    # 2) sort structures by increasing nonmetal counts (same behavior as your script)
    counts_sorted = sorted(counts, key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in counts_sorted]

    # 3) equal-ish split of μ-points among structures
    points_per_structure = n_grid // n_images
    remainder = n_grid % n_images

    # 4) Sort μ-grid by sum over the provided element columns
    sort_cols = sorted(set(element_column_map.values()))
    sort_key = np.sum(reaction_conditions[:, sort_cols], axis=1)
    grid_sorted = reaction_conditions[np.argsort(sort_key)]

    # 5) Allocate contiguous windows (low -> high) to structures in the order of sorted_indices
    start = 0
    for rank, img_idx in enumerate(sorted_indices):
        size = points_per_structure + (1 if rank < remainder else 0)
        if size <= 0:
            mu_window = np.empty((0, reaction_conditions.shape[1]))
            center_mu = np.full(reaction_conditions.shape[1], np.nan)
        else:
            mu_window = grid_sorted[start:start+size]
            start += size
            center_row = mu_window[len(mu_window)//2]
            center_mu = np.round(center_row, round_decimals)

        # set atoms.info['data'] entries in-place
        ai = atoms_list[img_idx]
        if not hasattr(ai, 'info') or ai.info is None:
            ai.info = {'key_value_pairs': {}, 'data': {}}
        if 'data' not in ai.info or ai.info['data'] is None:
            ai.info['data'] = {}
        ai.info['data']['mu'] = np.array(center_mu)
        ai.info['data']['mu_window'] = np.array(mu_window)

    return atoms_list


def _to_array_and_broadcast_single(dct):
    """Turn dict values (scalar or 1D array) to arrays and broadcast to common length L."""

    arrs = {}
    lengths = []
    for k, v in dct.items():
        a = np.asarray(v)
        if a.ndim == 0:
            lengths.append(1)
        elif a.ndim == 1:
            lengths.append(a.shape[0])
        else:
            raise ValueError(f"Input for '{k}' must be scalar or 1D array.")
        arrs[k] = a
    L = max(lengths)
    for k, a in arrs.items():
        if a.ndim == 0:
            arrs[k] = np.full(L, float(a))
        elif a.shape[0] == 1 and L > 1:
            arrs[k] = np.full(L, float(a[0]))
        elif a.shape[0] == L:
            arrs[k] = a.astype(float)
        else:
            raise ValueError(f"Cannot broadcast '{k}' length {a.shape[0]} to common length {L}.")

    return arrs, L


def get_elemental_mu_limits(mu_gas, det_tol=1e-12):
    """
    Compute min/max elemental chemical potentials from molecular chemical potentials.

    Parameters
    ----------
    mu_gas : dict
        keys = molecule formulas (strings, e.g. 'CO2','C3H6','H2O'), values = float or 1D np.array
        representing the absolute chemical potential mu_g (in eV). Arrays must have the same length.

    det_tol : float
        determinant tolerance used to consider a stoichiometric submatrix invertible.

    Returns
    -------
    out : dict
        keys: '<ELEMENT>_min', '<ELEMENT>_max' (e.g. 'C_min','C_max','O_min','O_max', ...)
        Values: float (if inputs scalar) or 1D np.array of length L (if inputs arrays).
    """

    # broadcast input arrays; all species will have same length L
    mu_arrs, L = _to_array_and_broadcast_single(mu_gas)
    # drop species that carry no information (all-NaN)
    mu_arrs = {mol: arr for mol, arr in mu_arrs.items() if np.any(np.isfinite(arr))}
    if len(mu_arrs) == 0:
        raise ValueError('All provided mu_gas values are NaN or empty. Provide at least one finite mu.')

    mol_list = list(mu_arrs.keys())
    n_species_total = len(mol_list)

    # global element set from species that remain (deterministic ordering)
    element_set = set()
    stoich_list_global = []
    for mol in mol_list:
        atom_list = list(Formula(mol))
        counts = {}
        for el in atom_list:
            counts[el] = counts.get(el, 0) + 1
            element_set.add(el)
        stoich_list_global.append(counts)
    element_list_global = sorted(element_set)

    # prepare outputs (per-element, scalars or arrays)
    if L == 1:
        elem_min = {el: None for el in element_list_global}
        elem_max = {el: None for el in element_list_global}
    else:
        elem_min = {el: np.full(L, np.nan) for el in element_list_global}
        elem_max = {el: np.full(L, np.nan) for el in element_list_global}

    # Main per-condition loop
    for idx in range(L):
        # pick species that have finite mu at this index
        available_species = [mol for mol in mol_list if np.isfinite(mu_arrs[mol][idx])]
        if len(available_species) == 0:
            # nothing to do: all elements remain NaN for this index
            continue

        # build stoichiometry A_idx and element_list_idx from available_species
        stoich_list_idx = []
        element_set_idx = set()
        for mol in available_species:
            atom_list = list(Formula(mol))
            counts = {}
            for el in atom_list:
                counts[el] = counts.get(el, 0) + 1
                element_set_idx.add(el)
            stoich_list_idx.append(counts)
        element_list_idx = sorted(element_set_idx)
        n_species = len(available_species)
        n_elems = len(element_list_idx)

        # build A_idx (n_species x n_elems)
        A_idx = np.zeros((n_species, n_elems), dtype=float)
        for i, counts in enumerate(stoich_list_idx):
            for j, el in enumerate(element_list_idx):
                A_idx[i, j] = float(counts.get(el, 0))

        # build molecular mu vector b_idx
        b_idx = np.array([mu_arrs[mol][idx] for mol in available_species], dtype=float)

        # if there are enough species to form square systems, enumerate combinations
        solutions = []
        if n_elems <= n_species:
            for combo in combinations(range(n_species), n_elems):
                A_sub = A_idx[list(combo), :]   # shape (n_elems x n_elems)
                if abs(np.linalg.det(A_sub)) <= det_tol:
                    continue
                b_sub = b_idx[list(combo)]
                try:
                    mu_elem = np.linalg.solve(A_sub, b_sub)   # length n_elems
                except np.linalg.LinAlgError:
                    continue
                solutions.append(mu_elem)

        if len(solutions) > 0:
            sols = np.vstack(solutions)  # (n_solutions, n_elems)
            mins = np.min(sols, axis=0)
            maxs = np.max(sols, axis=0)
            # assign mins/maxs into global element arrays (for elements missing here, leave NaN)
            for j, el in enumerate(element_list_idx):
                if L == 1:
                    elem_min[el] = float(mins[j])
                    elem_max[el] = float(maxs[j])
                else:
                    elem_min[el][idx] = float(mins[j])
                    elem_max[el][idx] = float(maxs[j])
        else:
            # fallback least-squares on available species
            warnings.warn('Degenerate case! Fall back to least-squares solutions with min = max = sol')
            try:
                sol, *_ = np.linalg.lstsq(A_idx, b_idx, rcond=None)
            except np.linalg.LinAlgError:
                sol = np.full(n_elems, np.nan)
            for j, el in enumerate(element_list_idx):
                if L == 1:
                    elem_min[el] = float(sol[j])
                    elem_max[el] = float(sol[j])
                else:
                    elem_min[el][idx] = float(sol[j])
                    elem_max[el][idx] = float(sol[j])

    # build output dict using global element list ordering
    out = {}
    for el in element_list_global:
        out[f'{el}_min'] = elem_min[el]
        out[f'{el}_max'] = elem_max[el]

    return out
