#!/usr/bin/env python
# -*- coding: utf-8 -*-
from acat.build.ordering import RandomOrderingGenerator as ROG
from acat.build.ordering import SymmetricSlabOrderingGenerator as SSOG
from acat.adsorption_sites import SlabAdsorptionSites
from acat.build.adlayer import min_dist_coverage_pattern as mindcp
from acat.build.adlayer import max_dist_coverage_pattern as maxdcp
from ase.io import read, write
from ase.formula import Formula
from collections import Counter
import numpy as np
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
    if set(template.symbols).isdisjoint(set(target_metal_species)) and target_metal_species:
        for atoms in template:
            atoms.symbol = target_metal_species[0] 
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
                ads_species = [adsorbate_species[i] for i in
                               np.random.choice(list(range(len(adsorbate_species))),
                               size=max_adsorbate_species, replace=False)]
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
    Assign mu-windows to a list of ase.Atoms objects using a balanced recursive
    partition of the full mu-grid.

    The algorithm:
    1) ranks replicas by their nonmetal composition;
    2) partitions reaction_conditions into disjoint windows with equal point counts;
    3) uses recursive median splits along the widest mu-dimension so each window
       is compact and as close to equal-volume as possible on a grid;
    4) assigns low-mu windows to low-nonmetal replicas and high-mu windows to
       high-nonmetal replicas.

    Parameters
    ----------
    input_pop : list of ase.Atoms
        Replicas to assign.
    reaction_conditions : np.ndarray
        Shape (n_points, n_cols). Each row is a mu-point.
    element_column_map : dict
        Mapping from non-metal element name to its mu-column index.
    round_decimals : int
        Number of decimals used when storing the representative mu center.

    """

    atoms_list = list(input_pop)
    n_images = len(atoms_list)
    if n_images == 0:
        return atoms_list
    reaction_conditions = np.asarray(reaction_conditions, dtype=float)
    if reaction_conditions.ndim != 2:
        raise ValueError('reaction_conditions must be a 2D array (n_points, n_cols)')
    n_grid = reaction_conditions.shape[0]
    if n_images > n_grid:
        raise ValueError(
            f'Cannot assign {n_images} replicas to only {n_grid} mu-points. '
            'Need n_images <= n_grid.'
        )

    # Nonmetal species ordered by their mu-column index.
    nonmetal_species = [k for k, _ in sorted(element_column_map.items(), key=lambda kv: kv[1])]
    mu_cols = np.array([element_column_map[k] for k in nonmetal_species], dtype=int)

    # Replica ordering: low nonmetal content -> low mu-window, high nonmetal content -> high mu-window.
    counts = []
    for i, atoms in enumerate(atoms_list):
        cnt_vec = np.array([sum(1 for a in atoms if a.symbol == sp) for sp in nonmetal_species], dtype=int)
        counts.append((i, cnt_vec))
    sorted_indices = [idx for idx, _ in sorted(counts, key=lambda x: tuple(x[1].tolist()))]

    # Use all reaction-condition columns to define compact windows in the full mu-space.
    mu_points = reaction_conditions[:, mu_cols]

    # Normalize each mu dimension for balanced geometric splitting.
    mu_min = mu_points.min(axis=0)
    mu_max = mu_points.max(axis=0)
    mu_span = np.where((mu_max - mu_min) > 0, (mu_max - mu_min), 1.0)
    mu_norm = (mu_points - mu_min) / mu_span

    # Desired sizes
    q, r = divmod(n_grid, n_images)
    target_sizes = [q + 1] * r + [q] * (n_images - r)

    def _partition(indices, sizes):
        """Recursively partition indices into disjoint groups with target sizes."""
        if len(sizes) == 1:
            return [np.asarray(indices, dtype=int)]

        pts = mu_norm[indices]
        spreads = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(spreads))

        # Lexsort with the chosen axis as primary key.
        keys = tuple(pts[:, d] for d in range(pts.shape[1]) if d != axis) + (pts[:, axis],)
        order = np.asarray(indices, dtype=int)[np.lexsort(keys)]

        left_n = int(sum(sizes[:len(sizes) // 2]))
        left_inds = order[:left_n]
        right_inds = order[left_n:]

        return _partition(left_inds, sizes[:len(sizes) // 2]) + _partition(right_inds, sizes[len(sizes) // 2:])

    # Partition the mu-grid.
    all_indices = np.arange(n_grid, dtype=int)
    groups = _partition(all_indices, target_sizes)

    # Sort groups by centroid in the same mu-order used for replica ordering.
    def _group_key(inds):
        c = mu_points[inds].mean(axis=0)
        return tuple(np.round(c, round_decimals).tolist())

    groups = sorted(groups, key=_group_key)

    # Assign windows to replicas.
    for rank, img_idx in enumerate(sorted_indices):
        inds = groups[rank]
        mu_window = reaction_conditions[inds]

        center_mu = np.round(mu_window.mean(axis=0), round_decimals)
        #mu_lower = np.round(mu_window.min(axis=0), round_decimals)
        #mu_upper = np.round(mu_window.max(axis=0), round_decimals)

        ai = atoms_list[img_idx]
        if not hasattr(ai, 'info') or ai.info is None:
            ai.info = {'key_value_pairs': {}, 'data': {}}
        if 'data' not in ai.info or ai.info['data'] is None:
            ai.info['data'] = {}

        ai.info['data']['mu'] = np.array(center_mu)
        ai.info['data']['mu_window'] = np.array(mu_window)
        #ai.info['data']['mu_lower'] = np.array(mu_lower)
        #ai.info['data']['mu_upper'] = np.array(mu_upper)

    return atoms_list


def get_ls_elemental_mu(mu_gas, elemental_mu=None):
    """
    Compute least-squares elemental chemical potentials from gas chemical potentials.

    Parameters
    ----------
    mu_gas: dict {str: float or 1D np.array}
        A dictionary that maps between gas molecule formula and absolute molecular chemical 
        potential(s) in eV.

    elemental_mu: dict {str: float or 1D np.array}, default None 
        Optional pinned gas molecule chemical potential(s) in eV.

    """

    # build stoichiometric matrix N (n_species x M)
    species = list(mu_gas.keys())
    counts = []
    elems_set = set()
    for s in species:
        atom_list = list(Formula(s))
        c = Counter(atom_list)
        counts.append(c)
        elems_set.update(c.keys())
    elems = sorted(elems_set)
    n_species = len(species)
    M = len(elems)
    N = np.array([[counts[i].get(el, 0) for el in elems] for i in range(n_species)], dtype=float)

    # broadcast molecular vector g into G (n_species x L)
    lengths = []
    for v in mu_gas.values():
        a = np.asarray(v)
        if a.ndim == 0:
            lengths.append(1)
        elif a.ndim == 1:
            lengths.append(a.shape[0])
        else:
            raise ValueError('mu_gas values must be scalar or 1D arrays.')
    L = int(max(lengths))
    G = np.zeros((n_species, L), dtype=float)
    for i, sp in enumerate(species):
        a = np.asarray(mu_gas[sp], dtype=float)
        if a.ndim == 0:
            G[i, :] = float(a)
        elif a.shape[0] == L:
            G[i, :] = a
        elif a.shape[0] == 1:
            G[i, :] = float(a[0])
        else:
            raise ValueError(f"Cannot broadcast mu_gas['{sp}'] to length {L}.")

    # process pinned elemental μ if provided
    pinned = {}
    if elemental_mu is not None:
        for el, val in elemental_mu.items():
            if el not in elems:
                raise ValueError(f"Pinned element '{el}' not present in parsed elements {elems}.")
            a = np.asarray(val, dtype=float)
            if a.ndim == 0:
                pinned[el] = np.full(L, float(a))
            elif a.ndim == 1 and a.shape[0] in (1, L):
                pinned[el] = a.copy() if a.shape[0] == L else np.full(L, float(a[0]))
            else:
                raise ValueError(f"Cannot broadcast elemental_mu['{el}'] to length {L}.")

    # partition pinned / free
    pinned_mask = np.array([el in pinned for el in elems], dtype=bool)
    free_idx = np.where(~pinned_mask)[0].tolist()
    pinned_idx = np.where(pinned_mask)[0].tolist()
    MU = np.zeros((M, L), dtype=float)
    # fill pinned into MU if present
    if pinned_idx:
        for j in pinned_idx:
            MU[j, :] = pinned[elems[j]]

    # if nothing free, just return pinned
    if len(free_idx) == 0:
        return {el: (float(MU[j, 0]) if L == 1 else MU[j, :].copy()) for j, el in enumerate(elems)}

    # RHS adjusted by pinned contribution (if any)
    if pinned_idx:
        N_p = N[:, pinned_idx]
        m_p = np.vstack([MU[j, :] for j in pinned_idx])   # n_pinned x L
        RHS = G - N_p.dot(m_p)
    else:
        RHS = G

    # LS solve for free variables: mu_free = pinv(N_free) @ RHS
    N_free = N[:, free_idx]
    mu_free = np.linalg.pinv(N_free).dot(RHS)   # n_free x L

    for k, j in enumerate(free_idx):
        MU[j, :] = mu_free[k, :]

    return {el: (float(MU[j, 0]) if L == 1 else MU[j, :].copy()) for j, el in enumerate(elems)}
