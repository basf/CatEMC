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
    Generate an EMC population of random bulk/slab configurations.

    Parameters
    ----------
    template: ase.Atoms object
        Template structure for generating the initial randomized EMC population.

    pop_size: int 
        The number of random configurations to generate in the population.

    target_metal_species: list of strs, default None
        A list of possible metal species to randomly present in the catalyst. 
        Default None will not randomize metal atoms.

    target_metal_composition: dict of (str, float) pairs, default None
        A dictionary that maps each metal species to the corresponding concentrations.
        Default None will not fix the composition when randomizing metal atoms.

    target_nonmetal_species: list or strs, default None 
        A list of possible non-metal species to randomly present in the catalyst.
        Default None will not randomize non-metal atoms.

    target_nonmetal_composition: dict of (str, float) pairs, default None
        A dictionary that maps each non-metal species to the corresponding 
        concentrations. Default None will not fix the composition when randomizing 
        non-metal atoms.

    allowed_indices: list of ints, default None
        A list of atom indices that are allowed to mutate in the randomizatino. 
        Default None will allow all atom indices to mutate.

    valences: dict of (str, int) pairs, default None
        A dictionary that maps the each species to the corresponding charge state 
        to ensure charge neutrality of the structures.

    adsorbate_species: list or strs, default None 
        A list of possible adsorbate species to randomly adsorb on the catalyst 
        surface. Default None will not add any adsorbate species.
        
    adsorbate_coverage: float, default None
        Fix to this surface coverage (in ML, divided by the number of surface metal 
        atoms) when adding adsorbate species.

    adsorbate_adsorbate_dmin: float, defeault None 
        The radius of the sphere inside which no other adsorbates should be found.

    max_adsorbate_species: int, default None 
        The maximum allowed types of adsorbate species (excluding vacancies) for a 
        single structure. Allow all adsorbatae species if not specified.

    adsorption_sites: acat.adsorption_sites.SlabAdsorptionSites object or the corresponding pickle file path, default None
        Provide the ACAT built-in adsorption sites class to accelerate the adlayer 
         pattern generation.

    adsorption_dict: dict, default {}
         ACAT SlabAdsorptionSites parameters for identifying adsorption sites.

    symmetry_constraint: bool, default False
        Whether to apply symmetry constraint. Currently only support generating 
        symmetric slabs.

    symmetry_dict: dict, default {}
        ACAT SymmetricSlabOrderingGenerator paramters for generating symmetric slabs.

    trajectory: str, default None 
        Path for saving the structures in the inital population. Default None will
        only return the list of structures but not writing them.

    seed: int, default None 
        Random seed for reproducibility.

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
            """
            if input_type == 'cluster':
                from acat.build.ordering import SymmetricClusterOrderingGenerator as SCOG
            elif input_type == 'slab':
                from acat.build.ordering import SymmetricSlabOrderingGenerator as SSOG
            """
            from acat.build.ordering import SymmetricSlabOrderingGenerator as SSOG
            symmetry_dict['atoms'] = metal_atoms
            symmetry_dict['elements'] = target_metal_species
            symmetry_dict['composition'] = target_metal_composition
            symmetry_dict['trajectory'] = 'metal_orderings.traj'
            symmetry_dict['save_groups'] = True
            gen = SSOG(**symmetry_dict)
            gen.run(max_gen=ngen)
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
    Assign mu-windows to a list of ase.Atoms objects by dividing the full
    mu-grid into regular blocks.

    The algorithm:

    1) ranks replicas by their nonmetal composition;
    2) partitions reaction_conditions into disjoint windows with equal point
       counts (the windows differ by one point when the grid does not divide
       evenly);
    3) cuts the grid into slabs along one mu-dimension and every slab into
       windows along the next one. The cuts follow the grid lines wherever
       the point counts allow it, so the windows are rectangular blocks
       when the grid divides evenly (e.g. 3 x 3 squares for 49 replicas on
       a 21 x 21 grid) and blocks with a stepped edge otherwise. Of the
       possible numbers of slabs and cutting orders, the one with the most
       compact windows is used;
    4) assigns low-mu windows to low-nonmetal replicas and high-mu windows to
       high-nonmetal replicas.

    Parameters
    ----------
    input_pop : list of ase.Atoms objects
        Replicas to assign.

    reaction_conditions : np.ndarray
        Shape (n_points, n_cols). Each row is a mu-point.

    element_column_map : dict of (str, int) pairs
        Mapping from non-metal element name to its mu-column index.

    round_decimals : int
        Number of decimals used when comparing the mu-window centers to
        order the windows. The starting chemical potential of each replica
        is the point of its window that is closest to the window center.

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

    # Window sizes: n_big windows of q + 1 points and the others of q points.
    q, n_big = divmod(n_grid, n_images)
    n_dims = mu_norm.shape[1]

    def _order(indices, axis):
        """Sort points along one axis. Within one grid line of that axis the
        points follow the other axes back and forth, so that consecutive
        points are neighbours on the grid."""
        indices = np.asarray(indices, dtype=int)
        pts = np.round(mu_norm[indices], 9)
        dims = [axis] + [d for d in range(n_dims) if d != axis]
        keys, flip = [], np.zeros(len(indices), dtype=int)
        for d in dims:
            levels = np.unique(pts[:, d], return_inverse=True)[1].ravel()
            keys.append(np.where(flip % 2 == 1, levels.max() - levels, levels))
            flip = flip + levels
        return indices[np.lexsort(tuple(reversed(keys)))]

    def _cut(indices, axis, k, a, m):
        """Cut a point set along one axis into m consecutive slabs.

        The set holds k windows, a of them with q + 1 points. Every slab
        receives a share of the windows (within one of k / m), and its
        number of larger windows is chosen so that the cut falls on a grid
        line whenever that is possible.
        """
        order = _order(indices, axis)
        coords = np.round(mu_norm[order, axis], 9)
        on_grid_line = set((np.flatnonzero(np.diff(coords) > 0) + 1).tolist())

        slabs = []
        start, k_rem, a_rem = 0, k, a
        for slabs_rem in range(m, 0, -1):
            if slabs_rem == 1:
                k_i, a_i = k_rem, a_rem
            else:
                best = None
                for k_i in sorted({k_rem // slabs_rem, -(-k_rem // slabs_rem)}):
                    a_min = max(0, a_rem - (k_rem - k_i))
                    a_max = min(k_i, a_rem)
                    for a_i in range(a_min, a_max + 1):
                        stop = start + k_i * q + a_i
                        # prefer a cut on a grid line, then even slabs
                        key = (stop not in on_grid_line,
                               abs(a_i * k_rem - a_rem * k_i),
                               abs(k_i * slabs_rem - k_rem))
                        if best is None or key < best[0]:
                            best = (key, k_i, a_i)
                _, k_i, a_i = best
            stop = start + k_i * q + a_i
            slabs.append((order[start:stop], k_i, a_i))
            start, k_rem, a_rem = stop, k_rem - k_i, a_rem - a_i
        return slabs

    def _scatter(windows):
        """Sum of squared distances to the window centres (compactness)."""
        return sum(float(((mu_norm[w] - mu_norm[w].mean(axis=0)) ** 2).sum())
                   for w in windows)

    def _partition(indices, k, a, axes):
        """Divide a point set into k windows, a of them with q + 1 points."""
        if k == 1:
            return [np.asarray(indices, dtype=int)]

        pts = mu_norm[indices]
        n_levels = {d: len(np.unique(np.round(pts[:, d], 9))) for d in axes}
        active = [d for d in axes if n_levels[d] > 1]
        if len(active) <= 1:
            # One direction left: one window after the other along it.
            axis = active[0] if active else int(np.argmax(np.ptp(pts, axis=0)))
            return [w for w, _, _ in _cut(indices, axis, k, a, k)]

        # Extent of the point set along every direction, including the
        # half grid spacing at both ends.
        extent = {d: np.ptp(pts[:, d]) * n_levels[d] / (n_levels[d] - 1)
                  for d in active}
        scale = (k / np.prod([extent[d] for d in active])) ** (1. / len(active))

        best = None
        for axis in active:
            others = [d for d in active if d != axis]
            # Number of slabs that makes the windows as wide as they are high.
            ideal = extent[axis] * scale
            lower = int(np.floor(ideal))
            most = min(k, n_levels[axis])
            for m in sorted({min(max(m, 1), most)
                             for m in range(lower - 1, lower + 3)}):
                windows = []
                for inds, k_i, a_i in _cut(indices, axis, k, a, m):
                    windows += _partition(inds, k_i, a_i, others)
                score = _scatter(windows)
                if best is None or score < best[0] - 1e-12:
                    best = (score, windows)
        return best[1]

    # Partition the mu-grid.
    all_indices = np.arange(n_grid, dtype=int)
    groups = _partition(all_indices, n_images, n_big, list(range(n_dims)))

    # Sort groups by their centre in the same mu-order used for replica
    # ordering. The centre is the middle of the range of the window in every
    # mu-dimension, so the windows of one slab share the first coordinate
    # exactly and are then ordered along the next one.
    def _group_key(inds):
        c = (mu_points[inds].min(axis=0) + mu_points[inds].max(axis=0)) / 2.
        return tuple(np.round(c, round_decimals).tolist())

    groups = sorted(groups, key=_group_key)

    # Assign windows to replicas.
    for rank, img_idx in enumerate(sorted_indices):
        inds = groups[rank]
        mu_window = reaction_conditions[inds]

        # The starting chemical potential must be a state of the window
        # itself (the mu-walk and the MBAR bias both look it up in the
        # window / reaction-condition grid), so the window point closest to
        # the centroid is used instead of the centroid, which in general
        # falls between grid points.
        centroid_dists = np.linalg.norm(mu_norm[inds] - mu_norm[inds].mean(axis=0), axis=1)
        center_mu = mu_window[int(np.argmin(centroid_dists))].copy()
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
