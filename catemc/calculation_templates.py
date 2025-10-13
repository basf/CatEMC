#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .gc_utils import compute_phi
from acat.adsorption_sites import SlabAdsorptionSites
from acat.adsorbate_coverage import SlabAdsorbateCoverage
from ase.io import write
from ase import units
from ase.neighborlist import NeighborList
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from ase.io.trajectory import Trajectory
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.parallel import parprint
import importlib
import importlib.util
import numpy as np


def calculate_bulk_fitness(
        atoms,
        traj_path,
        calculator_list,
        fmax_list,
        target_metal_species,
        run_md,
        md_timestep,
        md_taut,
        md_steps,
        md_write_interval,
        pressure=None,
        md_taup=1000*units.fs,
        fix_indices=None,
        update_md_positions=True,
        update_relaxed_positions=False,
        neighbor_cutoff=2.5,
        raw_score_cutoff=-10000.,
        temperature_key='temperature',
        replica_key='replica'):

    """
    A template function for calculating the fitness of a bulk 
    structure using some calculators in the GE-EMC simulation.

    Parameters
    ----------
    atoms: ase.Atoms
        Bulk structure to be calculated.

    traj_path: str
        The path to the atoms.traj file.

    calculator_list: list of ase.calculators.Calcuator objects
        All calculators that are used to evaluate the structure
        in sequence. The last calculator should be the main
        calculator to perform MD / energy minimization.

    fmax_list: list of floats/None
        The fmax convergence criteria for the relaxation using
        each calculator in the same order. Use None if you don't
        want to use that calculator to relax the structure.

    target_metal_species: list of strs
        Target metal elements in the bulk structure.

    run_md: bool
        Whether to run molecular dynamics or not.

    md_timestep, md_taut, md_steps, md_write_interval, md_taup: 
        MD parameters

    pressure: pressure, default None
        Target pressure for running NPT and cell relaxation. Running
        NVT MD and fixed-cell relaxation if this is not set.

    update_md_positions: bool, default True
        Whether to update the atom positions of the trajectory after
        molecular dynamics. Recommended to set to True.

    update_relaxed_positions: bool, default False
        Wheter to update the atom position of the trajectory after
        geometry relaxation. Recommended to set to False.

    neighbor_cutoff: float, default 2.5
        Cutoff (Å) for identify neighboring metal atoms to track
        vacancy movement.

    """

    assert len(calculator_list) == len(fmax_list), 'calculator_list and fmax_list must match length'
    dir_name = '/'.join(traj_path.split('/')[:-1])
    T = atoms.info['key_value_pairs'][temperature_key]
    rep = atoms.info['key_value_pairs'][replica_key]
    atoms = rattle_close_atoms(atoms, min_distance=0.1)
    md_atoms = None
    if fix_indices is None:
        fix_indices = []
    else:
        atoms.set_constraint(FixAtoms(indices=fix_indices))
    mut_metal_indices = [a.index for a in atoms if (a.index not in fix_indices) 
                         and (a.symbol in target_metal_species)]
    vac_indices = [a.index for a in atoms if a.symbol == 'X']
    real_indices = [a.index for a in atoms if a.symbol != 'X']    
    tmp_atoms = atoms[real_indices]

    for i, calc in enumerate(calculator_list):
        fmax = fmax_list[i]
        if i < len(calculator_list) - 1:
            try:
                tmp_atoms.calc = calc
                if fmax is not None:
                    opt = LBFGS(tmp_atoms, logfile=None)
                    opt.run(fmax=fmax, steps=250)
                E0 = tmp_atoms.get_potential_energy()
            except:
                tmp_atoms = atoms[real_indices]
                continue
            if np.isinf(E0):
                penalize = True
                break
        elif i == len(calculator_list) - 1:
            tmp_atoms.calc = calc
            E0 = tmp_atoms.get_potential_energy()
            if np.isinf(E0):
                penalize = True
                break
            if run_md:
                tag = f"{dir_name}/md"
                MaxwellBoltzmannDistribution(tmp_atoms, temperature_K=T, force_temp=True)
                Stationary(tmp_atoms)
                if pressure is None:                                                              
                    dyn = NVTBerendsen(tmp_atoms, timestep=md_timestep, temperature_K=T, 
                                       taut=md_taut, logfile=tag+f'_{rep}.log',
                                       trajectory=tag+f'_{rep}.traj', append_trajectory=True)
                else:
                    dyn = NPTBerendsen(tmp_atoms, timestep=md_timestep, temperature_K=T,
                                       taut=md_taut, pressure_au=pressure,
                                       taup=md_taup, compressibility_au=4.57e-5/units.bar,
                                       logfile=tag+f'_{rep}.log', trajectory=tag+f'_{rep}.traj', 
                                       append_trajectory=True)
                traj = Trajectory(tag+f'_{rep}.traj', 'a', tmp_atoms)
                dyn.attach(traj.write, interval=md_write_interval)
                dyn.run(md_steps)
                if update_md_positions:
                    md_atoms = tmp_atoms.copy()

            penalize = False
            if fmax is not None:
                if pressure is None:
                    opt = LBFGS(tmp_atoms, logfile=None)
                else:
                    ucf = FrechetCellFilter(tmp_atoms, scalar_pressure=pressure)
                    opt = LBFGS(ucf, logfile=None)
                converged = opt.run(fmax=fmax, steps=250)
                if not converged:
                    penalize = True

    epot = tmp_atoms.get_potential_energy()
    if penalize:
        pseudo_atoms = atoms.copy()
        pseudo_atoms.info['key_value_pairs']['raw_score'] = raw_score_cutoff
    else:
        if (md_atoms is not None) and update_md_positions and not update_relaxed_positions:
            pseudo_atoms = insert_vacancies(atoms, md_atoms, target_metal_species, neighbor_cutoff)
        elif update_relaxed_positions:
            pseudo_atoms = insert_vacancies(atoms, tmp_atoms, target_metal_species, neighbor_cutoff)
        else:
            pseudo_atoms = atoms.copy()
        pseudo_atoms.info['key_value_pairs']['raw_score'] = -epot

        if any(a for a in pseudo_atoms[mut_metal_indices] if a.symbol not in target_metal_species):
            parprint(f'Something wrong with replica {rep}, where metal indices are mutated into nonmetals')
            pseudo_atoms.info['key_value_pairs']['raw_score'] = raw_score_cutoff

    pseudo_atoms.info['key_value_pairs']['potential_energy'] = epot
    
    write(f"{dir_name}/relaxed_{traj_path.split('/')[-1]}", [atoms, pseudo_atoms])


def calculate_surface_fitness(
        atoms,
        traj_path,
        calculator_list,
        fmax_list,
        md_write_interval,
        target_metal_species,
        target_metal_composition,
        allowed_adsorbate_species,
        run_md,
        md_timestep,
        md_taut,
        md_steps,
        element_column_map,
        element_eref_map,
        element_y2x_map,
        adsorption_sites, 
        slab_indices, 
        fix_indices=None,
        temperature_dmu_map=None,
        update_md_positions=True,
        update_relaxed_positions=False,
        adsorbate_surface_dmax=2.5,
        neighbor_cutoff=2.5,
        raw_score_cutoff=-10000.,
        temperature_key='temperature',
        replica_key='replica'):

    """
    A template function for calculating the fitness of a surface 
    structure using some calculators in the GC-EMC simulation.

    Parameters
    ----------
    atoms: ase.Atoms
        Surface structure to be calculated.

    traj_path: str
        The path to the atoms.traj file.

    calculator_list: list of ase.calculators.Calcuator objects
        All calculators that are used to evaluate the structure
        in sequence. The last calculator should be the main
        calculator to perform MD / energy minimization.

    fmax_list: list of floats/None
        The fmax convergence criteria for the relaxation using
        each calculator in the same order. Use None if you don't
        want to use that calculator to relax the structure.

    target_metal_species: list of strs
        Target metal elements in the surface structure.

    target_metal_composition: dict of (str, float) pairs
        Target metal composition of the reference bulk structure.

    allowed_adsorbate_species: list of strs
        All possible adsorbate species allowed to exist in the EMC.

    run_md: bool
        Whether to run molecular dynamics or not.

    md_timestep, md_taut, md_steps, md_write_interval, md_taup: 
        MD parameters.

    pressure: float, default None
        Target pressure for running NPT and cell relaxation. Running
        NVT MD and fixed-cell relaxation if this is not set.

    update_md_positions: bool, default True
        Whether to update the atom positions of the trajectory after
        molecular dynamics. Recommended to set to True.

    update_relaxed_positions: bool, default False
        Wheter to update the atom position of the trajectory after
        geometry relaxation. Recommended to set to False.

    neighbor_cutoff: float, default 2.5
        Cutoff (Å) for identify neighboring metal atoms to track
        vacancy movement.

    """

    assert len(calculator_list) == len(fmax_list), 'calculator_list and fmax_list must match length'
    dir_name = '/'.join(traj_path.split('/')[:-1])
    T = atoms.info['key_value_pairs'][temperature_key]
    rep = atoms.info['key_value_pairs'][replica_key]
    sas = adsorption_sites
    mu = atoms.info['data']['mu']
    atoms = rattle_close_atoms(atoms, min_distance=0.1)
    md_atoms = None
    if fix_indices is None:
        fix_indices = []
    else:
        atoms.set_constraint(FixAtoms(indices=fix_indices))
    mut_metal_indices = [a.index for a in atoms if (a.index not in fix_indices) 
                         and (a.symbol in target_metal_species)]
    vac_indices = [a.index for a in atoms if a.symbol == 'X']
    real_indices = [a.index for a in atoms if a.symbol != 'X']    
    tmp_atoms = atoms[real_indices]
    tmp_slab_indices = [a.index for a in tmp_atoms if 
                        real_indices[a.index] in slab_indices]

    for i, calc in enumerate(calculator_list):
        fmax = fmax_list[i]
        if i < len(calculator_list) - 1:
            try:
                tmp_atoms.calc = calc
                if fmax is not None:
                    opt = LBFGS(tmp_atoms, logfile=None)
                    opt.run(fmax=fmax, steps=250)
                E0 = tmp_atoms.get_potential_energy()                
            except:
                tmp_atoms = atoms[real_indices]
                continue
            if np.isinf(E0):
                penalize = True
                break
        elif i == len(calculator_list) - 1:
            tmp_atoms.calc = calc
            E0 = tmp_atoms.get_potential_energy()
            if np.isinf(E0):
                penalize = True
                break
            if run_md:
                tag = f"{dir_name}/md"
                MaxwellBoltzmannDistribution(tmp_atoms, temperature_K=T, force_temp=True)
                Stationary(tmp_atoms)
                dyn = NVTBerendsen(tmp_atoms, timestep=md_timestep, temperature_K=T, 
                                   taut=md_taut, logfile=tag+f'_{rep}.log',
                                   trajectory=tag+f'_{rep}.traj', append_trajectory=True)
                                                                                                  
                traj = Trajectory(tag+f'_{rep}.traj', 'a', tmp_atoms)
                dyn.attach(traj.write, interval=md_write_interval)
                dyn.run(md_steps)
                if update_md_positions:
                    md_atoms = tmp_atoms.copy()

            penalize = False
            if fmax is not None:
                opt = LBFGS(tmp_atoms, logfile=None)
                converged = opt.run(fmax=fmax, steps=250)
                if not converged:
                    penalize = True
    if not penalize:
        if any(a for a in tmp_atoms if a.position[2] < atoms.positions[:,2].min()):
            penalize = True
        if any(a for a in tmp_atoms if (a.symbol in target_metal_species) and 
        (a.position[2] > atoms.positions[:,2].max() + adsorbate_surface_dmax)):
            penalize = True
        try:
            nsas = SlabAdsorptionSites(tmp_atoms, surface='autoadsorbate',
                                       composition_effect=False,
                                       ignore_sites='bridge')
            sas = nsas
        except:
            penalize = True
    if not penalize:
        sac = SlabAdsorbateCoverage(tmp_atoms, adsorption_sites=sas, 
                                    dmax=adsorbate_surface_dmax,
                                    adsorbate_indices=[a.index for a in tmp_atoms if 
                                                       a.index not in tmp_slab_indices])
        adsorbate_tuples = sac.get_adsorbates(allowed_adsorbate_species)
        adsorbates = [t[0] for t in adsorbate_tuples]
        adsorbate_ids = set() 
        rm_adsorbate_ids = set()
        for t in adsorbate_tuples:
            adsorbate_ids.update(t[1])
            if t[0] not in allowed_adsorbate_species:
                rm_adsorbate_ids.update(list(t[1]))
        adsorbate_indices = [real_indices[i] for i in sorted(adsorbate_ids - rm_adsorbate_ids)]
        new_atoms = tmp_atoms[[a.index for a in tmp_atoms if 
                               real_indices[a.index] in slab_indices+adsorbate_indices]]
        if len(new_atoms) < len(tmp_atoms):
            atoms = atoms[[a.index for a in atoms if a.index in slab_indices+adsorbate_indices]]
            real_indices = [a.index for a in atoms if a.symbol != 'X']
            vac_indices = [a.index for a in atoms if a.symbol == 'X']
            if fix_indices is not None:
                atoms.set_constraint(FixAtoms(indices=fix_indices))
            tmp_atoms = new_atoms.copy()
            tmp_atoms.calc = calc
            if fmax is not None:
                opt = LBFGS(tmp_atoms, logfile=None)
                converged = opt.run(fmax=fmax, steps=250)
                if not converged:
                    penalize = True
            sas.update(tmp_atoms)
            sac = SlabAdsorbateCoverage(tmp_atoms, adsorption_sites=sas, dmax=adsorbate_surface_dmax, 
                                        adsorbate_indices=[a.index for a in tmp_atoms if 
                                                           a.index not in tmp_slab_indices])
            adsorbate_tuples = sac.get_adsorbates(allowed_adsorbate_species)
            adsorbates = [t[0] for t in adsorbate_tuples]
            adsorbate_ids = set()
            for t in adsorbate_tuples:
                adsorbate_ids.update(t[1])
            adsorbate_indices = [real_indices[i] for i in sorted(adsorbate_ids)]

    epot = tmp_atoms.get_potential_energy()
    if penalize:
        pseudo_atoms = atoms.copy()
        pseudo_atoms.info['key_value_pairs']['raw_score'] = raw_score_cutoff
        fragments = [a.symbol for a in pseudo_atoms if a.symbol in element_column_map.keys()] 
    else:
        if (md_atoms is not None) and update_md_positions and not update_relaxed_positions:
            pseudo_atoms = insert_vacancies(atoms, md_atoms, target_metal_species, neighbor_cutoff)
        elif update_relaxed_positions:
            pseudo_atoms = insert_vacancies(atoms, tmp_atoms, target_metal_species, neighbor_cutoff)
        else:
            pseudo_atoms = atoms.copy()
        num_dict = dict((m, len([a for a in tmp_atoms if a.symbol==m])) 
                        for m in target_metal_species + list(element_column_map.keys()))
        num_metal = len([a for a in tmp_atoms if a.symbol in target_metal_species])

        phi = compute_phi(num_dict, target_metal_species, target_metal_composition,
                          temperature_dmu_map, np.array([np.append(mu, T)]), element_column_map, 
                          element_eref_map, element_y2x_map, num_metal, epot, dimensionless=False)[0]
        pseudo_atoms.info['key_value_pairs']['raw_score'] = -phi

        fragments = adsorbates + [a.symbol for a in tmp_atoms if (a.symbol in element_column_map.keys())
                                  and (real_indices[a.index] not in adsorbate_indices)]
        if any(a for a in pseudo_atoms[mut_metal_indices] if a.symbol not in target_metal_species):
            parprint(f'Something wrong with replica {rep}, where metal indices are mutated into nonmetals')
            pseudo_atoms.info['key_value_pairs']['raw_score'] = raw_score_cutoff
    pseudo_atoms.info['data']['fragments'] = fragments
    pseudo_atoms.info['key_value_pairs']['potential_energy'] = epot

    write(f"{dir_name}/relaxed_{traj_path.split('/')[-1]}", [atoms, pseudo_atoms])


def insert_vacancies(atoms, tmp_atoms, target_metal_species, neighbor_cutoff=2.5):
    """
    Fast centroid insertion with PBC-aware unwrapping (minimum-image 
    relative to vacancy) and guaranteed minimum distance.

    Parameters
    ----------
    atoms: ase.Atoms
        Original structure containing 'X' vacancy atoms.

    tmp_atoms: ase.Atoms
        Relaxed structure containing only the real atoms in the 
        same order as mask.

    target_metal_species: list of strs
        Symbols considered metal neighbors.

    neighbor_cutoff: float
        Cutoff (Å) for NeighborList on original topology.

    Returns
    -------
    pseudo_atoms: ase.Atoms
        Copy of `atoms` with real atoms from `tmp_atoms` and inserted 
        'X' positions.

    """

    # Indices mapping between original atoms and relaxed atoms
    vac_indices = [i for i, a in enumerate(atoms) if a.symbol == 'X']
    real_indices = [i for i, a in enumerate(atoms) if a.symbol != 'X']

    mask = np.ones(len(atoms), dtype=bool)
    mask[vac_indices] = False
    new_to_orig = np.arange(len(atoms))[mask].tolist()
    orig_to_new = [None] * len(atoms)
    for new_j, orig_i in enumerate(new_to_orig):
        orig_to_new[orig_i] = new_j

    # neighborlist built on original full-atom topology (to find which atoms neigh the vacancy)
    nl = NeighborList([neighbor_cutoff] * len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)

    # minimum allowed distance to any relaxed atom
    min_distance = 1.5
    eps = 1e-7

    # relaxed positions and cell for MIC unwrapping and vectorized distance
    R = tmp_atoms.get_positions()   # (N_real,3)
    cell = np.asarray(tmp_atoms.get_cell())   # (3,3)
    inv_cell = np.linalg.inv(cell)

    def min_image_min_dist_and_idx(candidate: np.ndarray):
        """
        Return (min_dist, idx, delta_cart) where delta_cart = vector from candidate to R[idx] image.
        Uses vectorized MIC math (no ASE calls).
        """
        if R.size == 0:
            return np.inf, None, None
        # vector from candidate to each relaxed atom
        delta = R - candidate                      # (N,3)
        delta_frac = delta @ inv_cell              # fractional differences
        delta_frac -= np.round(delta_frac)         # wrap to [-0.5,0.5)
        delta_cart = delta_frac @ cell             # (N,3) - vector from candidate to image of each atom
        d2 = np.einsum('ij,ij->i', delta_cart, delta_cart)
        idx = int(np.argmin(d2))
        min_d = float(np.sqrt(d2[idx]))
        return min_d, idx, delta_cart[idx]

    vac_positions = []
    # Precompute fractional coordinates of relaxed atoms (R) in tmp_atoms cell
    # (not strictly required for centroid, but convenient if needed later)
    # R_frac = R @ inv_cell
    for vi in vac_indices:
        neigh_inds, _ = nl.get_neighbors(vi)
        # collect mapped relaxed neighbor positions (Cartesian)
        neigh_pos = []
        for mi in neigh_inds:
            if atoms[mi].symbol in target_metal_species:
                mapped = orig_to_new[mi]
                if mapped is not None:
                    neigh_pos.append(R[mapped])
        if len(neigh_pos) == 0:
            # no metal neighbors -> keep the original vacancy location
            vac_positions.append(atoms.positions[vi].copy())
            continue

        neigh_pos = np.asarray(neigh_pos)   # (M,3)
        # Compute centroid using MIC relative to the vacancy position
        # Convert both neighbor positions and the vacancy position into fractional coords
        # using the relaxed cell (tmp_atoms cell), then wrap neighbor fractional differences
        # to [-0.5,0.5) relative to vacancy fractional coordinate, then convert back.
        vac_frac = (atoms.positions[vi] @ inv_cell)
        neigh_frac = neigh_pos @ inv_cell           
        # fractional difference from vacancy to each neighbor, wrapped to [-0.5,0.5)
        delta_frac = neigh_frac - vac_frac           
        delta_frac -= np.round(delta_frac)           
        neigh_images_frac = vac_frac + delta_frac    
        neigh_images_cart = neigh_images_frac @ cell 
        centroid = np.mean(neigh_images_cart, axis=0)
        # check MIC min distance from centroid to relaxed atoms (R)
        min_d, idx, delta_vec = min_image_min_dist_and_idx(centroid)
        if min_d >= min_distance - eps:
            vac_positions.append(centroid)
            continue

        # otherwise, nudge centroid away from nearest atom image so distance == min_distance
        # delta_vec is vector from centroid to that atom image (i.e. nearest_image_pos - centroid)
        # nearest_image_pos = centroid + delta_vec
        nearest_image_pos = centroid + delta_vec
        vec = centroid - nearest_image_pos   # vector from nearest atom to centroid
        norm = np.linalg.norm(vec)
        if norm < eps:
            # deterministic fallback direction to avoid zero vector
            unit = np.array([1.0, 0.0, 0.0])
        else:
            unit = vec / norm
        # place candidate at nearest_image_pos + unit * (min_distance + tiny)
        candidate = nearest_image_pos + unit * (min_distance + eps)
        # if candidate is still too close (very rare), fallback to original vacancy position
        min_d2, _, _ = min_image_min_dist_and_idx(candidate)
        if min_d2 < min_distance - eps:
            candidate = atoms.positions[vi].copy()

        vac_positions.append(candidate)

    # assemble pseudo_atoms: copy original, replace real atom positions by relaxed positions,
    # and put computed vacancy 'X' positions
    pseudo_atoms = atoms.copy()
    if len(real_indices) != 0:
        pseudo_atoms.positions[real_indices] = R
    if vac_indices:
        pseudo_atoms.positions[vac_indices] = np.vstack(vac_positions)

    return pseudo_atoms


def rattle_close_atoms(atoms, min_distance=0.1):

    if min_distance <= 0.0:
        return atoms

    D = atoms.get_all_distances(mic=True)
    np.fill_diagonal(D, np.inf)             
    i, j = np.where(D < float(min_distance))
    if i.size == 0:
        return atoms

    rattled_idx = np.unique(np.concatenate((i, j))).astype(int)
    sub = atoms[rattled_idx].copy()
    sub.rattle(stdev=float(min_distance) / 2.0)
    pos = atoms.get_positions()
    pos[rattled_idx] = sub.get_positions()
    atoms.set_positions(pos)

    return atoms


def detect_and_get_model(calc):
    """
    Exclusive detection of a short whitelist of MLIP calculators.
    Returns (True, model_obj) when `calc` is recognized and the expected
    attribute exists (non-None). Otherwise returns (False, None).

    Rules:
      - If `mace.calculators.mace` is available, treat any instance of
        MACECalculator (including user subclasses) as supported and prefer
        attributes 'mace_model' then 'model'. If none present, return the
        wrapper instance itself.
      - For other supported wrappers (fairchem, sevenn, orb_models,
        mattersim, graph_pes, metatomic), match by the exact module substring
        provided and return the first non-None attribute in the listed order.
      - No other calculators will be reported as supported.
    """
    import torch
    SUPPORTED = {
        'fairchem.core.calculate.ase_calculator': ('FAIRChemCalculator', ('predictor', 'model')),
        'sevenn.calculator': ('SevenNetCalculator', ('model',)),
        'orb_models.forcefield.calculator': ('ORBCalculator', ('model',)),
        'mattersim.forcefield.potential': ('MatterSimCalculator', ('model',)),
        'graph_pes.utils.calculator': ('GraphPESCalculator', ('model',)),
        'metatomic_torch.metatomic.torch.ase_calculator': ('MetatomicCalculator', ('model',)),
    }
    PREF_ATTRS_FOR_MACE = ('mace_model', 'model')

    mod = getattr(calc.__class__, '__module__', '') or ''
    name = getattr(calc.__class__, '__name__', '') or ''
    mod_l = mod.lower()
    name_l = name.lower()

    # 1) explicit, reliable MACE subclass detection (covers user subclasses)
    if importlib.util.find_spec('mace.calculators.mace') is not None:
        mace_mod = importlib.import_module('mace.calculators.mace')
        MACECalculator = getattr(mace_mod, 'MACECalculator', None)
        if MACECalculator is not None and isinstance(calc, MACECalculator):
            from torch_sim.models.mace import MaceModel
            ts_model = MaceModel(
                    model=calc.models[0],
                    device=torch.device(calc.device),
                    dtype=getattr(torch, calc.default_dtype),
                    compute_forces=True,
                    compute_stress=True,
                    enable_cueq=calc.enable_cueq)
            return True, ts_model 

    # 2) other explicit supported wrappers: match module substring OR class name,
    #    and return the first expected attribute that is non-None
    for mod_substr, (_cls_name, attrs) in SUPPORTED.items():
        if (mod_substr.lower() in mod_l) or (_cls_name.lower() == name_l):
            for a in attrs:
                model_obj = getattr(calc, a, None)
                if model_obj is not None:
                    return True, model_obj
            # matched wrapper identity but expected attributes missing -> not supported
            return False, None

    # 3) not supported
    return False, None


def _relax_worker(args):
    (tmp_atoms, calc, fmax, is_last_calc, run_md_flag, md_timestep, md_taut, md_steps, 
     md_write_interval, pressure, md_taup, temperature) = args
    tmp_atoms.calc = calc
    if is_last_calc and run_md_flag:
        # run MD as in serial function
        tag = 'md_worker'
        MaxwellBoltzmannDistribution(tmp_atoms, temperature_K=temperature, force_temp=True)
        Stationary(tmp_atoms)
        if pressure is None:
            dyn = NVTBerendsen(tmp_atoms, timestep=md_timestep, temperature_K=temperature,
                               taut=md_taut, logfile=None, trajectory=None, append_trajectory=False)
        else:
            dyn = NPTBerendsen(tmp_atoms, timestep=md_timestep, temperature_K=temperature,
                               taut=md_taut, pressure_au=pressure,
                               taup=md_taup, compressibility_au=4.57e-5/units.bar,
                               logfile=None, trajectory=None, append_trajectory=False)
        dyn.run(md_steps)
    if fmax is not None:
        # perform LBFGS relaxation like serial code
        opt = LBFGS(tmp_atoms, logfile=None)
        opt.run(fmax=fmax, steps=250)

    return tmp_atoms


def _postprocess_bulk_worker(args):
    (dir_name, orig_atoms, tmp_atoms, target_metal_species, fix_indices, neighbor_cutoff, run_md, 
     update_md_positions, update_relaxed_positions, raw_score_cutoff, replica_key, temperature_key, epot) = args

    penalize = False
    if epot is None or np.isinf(epot):
        penalize = True

    if penalize:
        pseudo_atoms = orig_atoms.copy()
        pseudo_atoms.info['key_value_pairs']['raw_score'] = raw_score_cutoff
    else:
        if (run_md and update_md_positions) and not update_relaxed_positions:
            pseudo_atoms = insert_vacancies(orig_atoms, tmp_atoms, target_metal_species, neighbor_cutoff)
        elif update_relaxed_positions:
            pseudo_atoms = insert_vacancies(orig_atoms, tmp_atoms, target_metal_species, neighbor_cutoff)
        else:
            pseudo_atoms = orig_atoms.copy()
        pseudo_atoms.info['key_value_pairs']['raw_score'] = -epot

        mut_metal_indices = [a.index for a in pseudo_atoms if (a.index not in (fix_indices or [])) and (a.symbol in target_metal_species)]
        if any(a for a in pseudo_atoms[mut_metal_indices] if a.symbol not in target_metal_species):
            parprint(f"Something wrong with replica {orig_atoms.info['key_value_pairs'][replica_key]}")
            pseudo_atoms.info['key_value_pairs']['raw_score'] = raw_score_cutoff

    pseudo_atoms.info['key_value_pairs']['potential_energy'] = epot if epot is not None else raw_score_cutoff
    write(f"{dir_name}/relaxed_atoms_{orig_atoms.info['key_value_pairs'][replica_key]}.traj", [orig_atoms, pseudo_atoms])

    return None


def _postprocess_surface_worker(args):
    (dir_name, orig_atoms, tmp_atoms, sas, target_metal_species, target_metal_composition,
     allowed_adsorbate_species, element_column_map, element_eref_map, element_y2x_map,
     slab_indices, neighbor_cutoff, run_md, update_md_positions, update_relaxed_positions,
     adsorbate_surface_dmax, raw_score_cutoff, temperature_dmu_map, temperature_key, replica_key, epot) = args

    penalize = False
    if epot is None or np.isinf(epot):
        penalize = True

    if not penalize:
        if any(a for a in tmp_atoms if a.position[2] < orig_atoms.positions[:, 2].min()):
            penalize = True
        if any(a for a in tmp_atoms if (a.symbol in target_metal_species) and
               (a.position[2] > orig_atoms.positions[:, 2].max() + adsorbate_surface_dmax)):
            penalize = True

    fragments = []
    if not penalize:
        nsas = SlabAdsorptionSites(tmp_atoms, surface='autoadsorbate', composition_effect=False, ignore_sites='bridge')
        sac = SlabAdsorbateCoverage(tmp_atoms, adsorption_sites=nsas,
                                    dmax=adsorbate_surface_dmax,
                                    adsorbate_indices=[a.index for a in tmp_atoms if a.index not in slab_indices])
        adsorbate_tuples = sac.get_adsorbates(allowed_adsorbate_species)
        adsorbates = [t[0] for t in adsorbate_tuples]
        adsorbate_ids = set(); rm_adsorbate_ids = set()
        for t in adsorbate_tuples:
            adsorbate_ids.update(t[1])
            if t[0] not in allowed_adsorbate_species:
                rm_adsorbate_ids.update(list(t[1]))
        adsorbate_indices = [i for i in sorted(adsorbate_ids - rm_adsorbate_ids)]

        # possibly reduce to slab+adsorbates and run a quick relaxation with final calc is handled by caller
        new_atoms = tmp_atoms[[a.index for a in tmp_atoms if a.index in (slab_indices + adsorbate_indices)]]
        if len(new_atoms) < len(tmp_atoms):
            tmp_atoms = new_atoms.copy()

    if penalize:
        pseudo_atoms = orig_atoms.copy()
        pseudo_atoms.info['key_value_pairs']['raw_score'] = raw_score_cutoff
        fragments = [a.symbol for a in pseudo_atoms if a.symbol in element_column_map.keys()]
    else:
        num_dict = dict((m, len([a for a in tmp_atoms if a.symbol == m]))
                        for m in target_metal_species + list(element_column_map.keys()))
        num_metal = len([a for a in tmp_atoms if a.symbol in target_metal_species])
        mu = orig_atoms.info['data']['mu']
        phi = compute_phi(num_dict, target_metal_species, target_metal_composition,
                          temperature_dmu_map, np.array([np.append(mu, orig_atoms.info['key_value_pairs'].get(temperature_key, None))]),
                          element_column_map, element_eref_map, element_y2x_map, num_metal, epot, dimensionless=False)[0]

        if (run_md and update_md_positions) and not update_relaxed_positions:
            pseudo_atoms = insert_vacancies(orig_atoms, tmp_atoms, target_metal_species, neighbor_cutoff)
        elif update_relaxed_positions:
            pseudo_atoms = insert_vacancies(orig_atoms, tmp_atoms, target_metal_species, neighbor_cutoff)
        else:
            pseudo_atoms = orig_atoms.copy()

        pseudo_atoms.info['key_value_pairs']['raw_score'] = -phi
        fragments = adsorbates + [a.symbol for a in tmp_atoms if (a.symbol in element_column_map.keys())]

        if any(a for a in pseudo_atoms if (a.symbol in target_metal_species) and
               (a.position[2] > orig_atoms.positions[:, 2].max() + adsorbate_surface_dmax)):
            parprint(f"Something wrong with replica {orig_atoms.info['key_value_pairs'][replica_key]}")
            pseudo_atoms.info['key_value_pairs']['raw_score'] = raw_score_cutoff

    pseudo_atoms.info['data']['fragments'] = fragments
    pseudo_atoms.info['key_value_pairs']['potential_energy'] = epot if epot is not None else raw_score_cutoff
    write(f"{dir_name}/relaxed_atoms_{orig_atoms.info['key_value_pairs'][replica_key]}.traj", [orig_atoms, pseudo_atoms])

    return None


def calculate_batch_bulk_fitness(
        atoms_list,
        traj_path,
        calculator_list,
        fmax_list,
        target_metal_species,
        run_md,
        md_timestep,
        md_taut,
        md_steps,
        md_write_interval,
        pressure=None,
        md_taup=1000 * units.fs,
        fix_indices=None,
        update_md_positions=True,
        update_relaxed_positions=False,
        neighbor_cutoff=2.5,
        raw_score_cutoff=-10000.,
        temperature_key='temperature',
        replica_key='replica',
        temperature_dmu_map=None):

    import torch_sim as ts
    import multiprocessing

    dir_name = '/'.join(traj_path.split('/')[:-1])
    # prepare tmp_atoms_list and bookkeeping
    tmp_atoms_list = []
    temps = []
    for orig_atoms in atoms_list:
        atoms = rattle_close_atoms(orig_atoms.copy(), min_distance=0.1)
        if fix_indices:
            atoms.set_constraint(FixAtoms(indices=fix_indices))
        real_indices = [a.index for a in atoms if a.symbol != 'X']
        tmp_atoms = atoms[real_indices].copy()
        tmp_atoms_list.append(tmp_atoms)
        temps.append(atoms.info['key_value_pairs'][temperature_key])

    last_relaxed_state = None
    # iterate calculators in sequence
    for idx, (calc, fmax) in enumerate(zip(calculator_list, fmax_list)):
        supported, ts_model = detect_and_get_model(calc)
        is_last = (idx == len(calculator_list) - 1)

        if supported and (fmax is not None) and (ts_model is not None):
            # MD via torch-sim: choose integrator based on presence of pressure
            if run_md:
                timestep_ps = float(md_timestep / units.fs) * 1e-3
                tag = f'{dir_name}/md'
                if pressure is None:
                    # NVT Langevin for pressure=None
                    integ_kwargs = dict(
                        system=tmp_atoms_list,
                        model=ts_model,
                        n_steps=int(md_steps),
                        temperature=temps,
                        timestep=timestep_ps,
                        integrator=ts.integrators.nvt_langevin,
                        autobatcher=True,
                        trajectory_reporter=dict(filenames=[f'{tag}_{i}.traj' for i in range(len(tmp_atoms_list))],
                                                 state_frequency=int(md_write_interval)),
                    )
                else:
                    # NPT Langevin for pressure != None (bulk)
                    integ_kwargs = dict(
                        system=tmp_atoms_list,
                        model=ts_model,
                        n_steps=int(md_steps),
                        temperature=temps,
                        timestep=timestep_ps,
                        integrator=ts.integrators.npt_langevin,
                        pressure=pressure,
                        autobatcher=True,
                        trajectory_reporter=dict(filenames=[f'{tag}_{i}.traj' for i in range(len(tmp_atoms_list))],
                                                 state_frequency=int(md_write_interval)),
                    )
                ts.integrate(**integ_kwargs)

            # Optimization via torch-sim: use FIRE; when pressure provided use Frechet cell filter via init_kwargs
            conv = ts.generate_force_convergence_fn(force_tol=fmax, include_cell_forces=False)
            if pressure is None:
                last_relaxed_state = ts.optimize(
                    system=tmp_atoms_list,
                    model=ts_model,
                    optimizer=ts.Optimizer.fire,
                    convergence_fn=conv,
                    autobatcher=True,
                )
            else:
                # Frechet cell + applied pressure for cell relaxation
                last_relaxed_state = ts.optimize(
                    system=tmp_atoms_list,
                    model=ts_model,
                    optimizer=ts.Optimizer.fire,
                    convergence_fn=conv,
                    autobatcher=True,
                    init_kwargs=dict(cell_filter=ts.CellFilter.frechet, pressure=pressure),
                )

            if hasattr(last_relaxed_state, 'to_atoms'):
                tmp_atoms_list = last_relaxed_state.to_atoms()
            elif isinstance(last_relaxed_state, (list, tuple)):
                tmp_atoms_list = list(last_relaxed_state)

        else:
            # Fallback to ASE LBFGS
            if fmax is None and not is_last:
                continue
            ctx = multiprocessing.get_context('spawn')
            pool = ctx.Pool(min(len(tmp_atoms_list), multiprocessing.cpu_count()))
            work_args = []
            for tmp, temp in zip(tmp_atoms_list, temps):
                work_args.append((tmp.copy(), calc, fmax, is_last, run_md, md_timestep, md_taut,
                                 md_steps, md_write_interval, pressure, md_taup, temp))
            results = pool.map(_relax_worker, work_args)
            pool.close()
            pool.join()
            tmp_atoms_list = results

    # try to extract energies from last_relaxed_state
    energies = None
    if last_relaxed_state is not None:
        energies = getattr(last_relaxed_state, 'energies', None) or getattr(last_relaxed_state, 'energy', None)

    # final_calc for ASE energy evaluation preference
    final_calc = calculator_list[-1]

    # assemble postprocessing args and run postprocessing in parallel
    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(min(len(atoms_list), multiprocessing.cpu_count()))
    post_args = []
    for orig_atoms, tmp_atoms in zip(atoms_list, tmp_atoms_list):
        # determine epot: ASE final_calc preferred
        epot = None
        if hasattr(final_calc, 'get_potential_energy') or hasattr(final_calc, 'implemented_properties'):
            tmp_atoms.calc = final_calc
            epot = tmp_atoms.get_potential_energy()
        # fallback to energies from TS state if present
        if epot is None and energies is not None:
            if hasattr(energies, '__len__'):
                if len(energies) > 0:
                    epot = float(energies[0])
                    try:
                        energies = energies[1:]
                    except Exception:
                        pass
            else:
                epot = float(energies)
        post_args.append((dir_name, orig_atoms, tmp_atoms, target_metal_species, fix_indices, neighbor_cutoff, run_md,
                          update_md_positions, update_relaxed_positions, raw_score_cutoff, replica_key, temperature_key, epot))

    pool.map(_postprocess_bulk_worker, post_args)
    pool.close()
    pool.join()


def calculate_batch_surface_fitness(
        atoms_list,
        traj_path,
        calculator_list,
        fmax_list,
        target_metal_species,
        target_metal_composition,
        allowed_adsorbate_species,
        run_md,
        md_timestep,
        md_taut,
        md_steps,
        md_write_interval,
        element_column_map,
        element_eref_map,
        element_y2x_map,
        adsorption_sites_list,
        slab_indices,
        fix_indices=None,
        temperature_dmu_map=None,
        update_md_positions=True,
        update_relaxed_positions=False,
        adsorbate_surface_dmax=2.5,
        neighbor_cutoff=2.5,
        raw_score_cutoff=-10000.,
        temperature_key='temperature',
        replica_key='replica'):

    import torch_sim as ts 
    import multiprocessing

    dir_name = '/'.join(traj_path.split('/')[:-1])
    # prepare tmp_atoms_list and bookkeeping
    tmp_atoms_list = []
    temps = []
    for orig_atoms in atoms_list:
        atoms = rattle_close_atoms(orig_atoms.copy(), min_distance=0.1)
        if fix_indices:
            atoms.set_constraint(FixAtoms(indices=fix_indices))
        real_indices = [a.index for a in atoms if a.symbol != 'X']
        tmp_atoms = atoms[real_indices].copy()
        tmp_atoms_list.append(tmp_atoms)
        temps.append(atoms.info['key_value_pairs'][temperature_key])

    last_relaxed_state = None
    # iterate calculators in sequence
    for idx, (calc, fmax) in enumerate(zip(calculator_list, fmax_list)):
        supported, ts_model = detect_and_get_model(calc)
        is_last = (idx == len(calculator_list) - 1)
        
        if supported and (fmax is not None) and (ts_model is not None):
            if run_md:
                timestep_ps = float(md_timestep / units.fs) * 1e-3
                tag = f'{dir_name}/md'
                # Surface MD: use NVT Langevin (no pressure)
                integ_kwargs = dict(
                    system=tmp_atoms_list,
                    model=ts_model,
                    n_steps=int(md_steps),
                    temperature=temps,
                    timestep=timestep_ps,
                    integrator=ts.integrators.nvt_langevin,
                    autobatcher=True,
                    trajectory_reporter=dict(filenames=[f'{tag}_{i}.traj' for i in range(len(tmp_atoms_list))],
                                             state_frequency=int(md_write_interval)),
                )
                ts.integrate(**integ_kwargs)
 
            # batched optimization for surface: FIRE (pressure not applicable)
            conv = ts.generate_force_convergence_fn(force_tol=fmax, include_cell_forces=False)
            last_relaxed_state = ts.optimize(
                system=tmp_atoms_list,
                model=ts_model,
                optimizer=ts.Optimizer.fire,
                convergence_fn=conv,
                autobatcher=True,
            )
            if hasattr(last_relaxed_state, 'to_atoms'):
                tmp_atoms_list = last_relaxed_state.to_atoms()
            elif isinstance(last_relaxed_state, (list, tuple)):
                tmp_atoms_list = list(last_relaxed_state)

        else:
            # Fallback to ASE LBFGS
            if fmax is None and not is_last:
                continue
            ctx = multiprocessing.get_context('spawn')
            pool = ctx.Pool(min(len(tmp_atoms_list), multiprocessing.cpu_count()))
            work_args = []
            for tmp, temp in zip(tmp_atoms_list, temps):
                work_args.append((tmp.copy(), calc, fmax, is_last, run_md, md_timestep,
                                  md_taut, md_steps, md_write_interval, None, None, temp))
            results = pool.map(_relax_worker, work_args)
            pool.close()
            pool.join()
            tmp_atoms_list = results

    # try to read energies from last_relaxed_state
    energies = None
    if last_relaxed_state is not None:
        energies = getattr(last_relaxed_state, 'energies', None) or getattr(last_relaxed_state, 'energy', None)

    final_calc = calculator_list[-1]

    # postprocess per-replica in parallel (adsorbate detection, phi, vacancy insertion, write)
    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(min(len(atoms_list), multiprocessing.cpu_count()))
    post_args = []
    for orig_atoms, tmp_atoms, sas in zip(atoms_list, tmp_atoms_list, adsorption_sites_list):
        epot = None
        if hasattr(final_calc, 'get_potential_energy') or hasattr(final_calc, 'implemented_properties'):
            tmp_atoms.calc = final_calc
            epot = tmp_atoms.get_potential_energy()
        if epot is None and energies is not None:
            if hasattr(energies, '__len__'):
                if len(energies) > 0:
                    epot = float(energies[0])
                    try:
                        energies = energies[1:]
                    except Exception:
                        pass
            else:
                epot = float(energies)
        post_args.append((dir_name, orig_atoms, tmp_atoms, sas, target_metal_species, target_metal_composition,
                          allowed_adsorbate_species, element_column_map, element_eref_map, element_y2x_map,
                          slab_indices, neighbor_cutoff, run_md, update_md_positions, update_relaxed_positions,
                          adsorbate_surface_dmax, raw_score_cutoff, temperature_dmu_map, temperature_key, replica_key, epot))
    pool.map(_postprocess_surface_worker, post_args)
    pool.close()
    pool.join()
