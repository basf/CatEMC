#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .gc_utils import compute_phi
from acat.adsorption_sites import SlabAdsorptionSites
from acat.adsorbate_coverage import SlabAdsorbateCoverage
from ase import units
from ase import optimize as ase_optimize
from ase.neighborlist import NeighborList
from ase.constraints import FixAtoms
from .optimizers import SQNM
from ase.filters import FrechetCellFilter
from ase.io.trajectory import Trajectory
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.parallel import parprint
import numpy as np
import copy


def _get_optimizer(optimizer):
    if optimizer == 'SQNM':
        return SQNM
    return getattr(ase_optimize, optimizer)


def calculate_bulk_fitness(
        atoms,
        calculator_list,
        fmax_list,
        md_timestep,
        md_taut,
        md_steps,
        md_write_interval,
        optimizer='SQNM',
        run_md=False,
        md_type='berendsen',
        md_friction=None,
        md_taup=1000*units.fs,
        compressibility=5e-5/units.bar,
        max_relax_steps=300,
        write_relax_traj=False,
        write_md_traj=False,
        optimize_cell=False,
        update_md_positions=True,
        update_relaxed_positions=False,
        neighbor_cutoff=2.5,
        _emc_context=None):

    """Calculate the fitness of a bulk EMC candidate.

    Parameters
    ----------
    atoms : ase.Atoms
        Bulk structure to evaluate.
    calculator_list : list of ase.calculators.calculator.Calculator
        Calculators applied sequentially. The final calculator is used for
        molecular dynamics, the final relaxation and the reported energy.
    fmax_list : list of float or None
        Force-convergence thresholds corresponding to ``calculator_list``.
        A value of ``None`` skips relaxation with that calculator.
    md_timestep : float
        Molecular-dynamics time step in ASE time units.
    md_taut : float
        Temperature-coupling time constant in ASE time units.
    md_steps : int
        Number of molecular-dynamics steps.
    md_write_interval : int
        Interval between frames in the optional molecular-dynamics trajectory.
    optimizer : str, default 'SQNM'
        Structure optimizer. ``'SQNM'`` uses :class:`catemc.optimizers.SQNM`;
        other names are resolved from :mod:`ase.optimize`.
    run_md : bool, default False
        Run molecular dynamics before the final relaxation.
    md_type : {"berendsen", "langevin"}, default "berendsen"
        Molecular-dynamics thermostat.
    md_friction : float or None, default None
        Langevin friction coefficient. Used only when ``md_type="langevin"``.
    md_taup : float, default 1000 * ase.units.fs
        Pressure-coupling time constant for NPT molecular dynamics.
    compressibility : float, default 5e-5 / ase.units.bar
        Isothermal compressibility in ASE units for NPT molecular dynamics.
    max_relax_steps : int, default 300
        Maximum number of optimization steps for each relaxation.
    write_relax_traj : bool, default False
        Write temporary relaxation trajectories.
    write_md_traj : bool, default False
        Write temporary molecular-dynamics trajectories and logs.
    optimize_cell : bool, default False
        Relax the cell together with atomic positions when a target pressure is
        defined by the EMC calculation.
    update_md_positions : bool, default True
        Use the post-MD coordinates when constructing the returned candidate.
    update_relaxed_positions : bool, default False
        Use the fully relaxed coordinates in the returned candidate. Otherwise,
        relaxation is used only to evaluate its fitness.
    neighbor_cutoff : float, default 2.5
        Neighbor cutoff in angstrom used to map vacancies after atomic motion.
    _emc_context : dict or None, default None
        Internal EMC state supplied by the simulation driver. It contains the
        thermodynamic, indexing and output settings that are intentionally not
        part of ``fitness_kwargs``.

    Returns
    -------
    ase.Atoms
        Evaluated candidate with its potential energy, volume and raw score.
    """

    if _emc_context is None:
        raise ValueError('Missing EMC fitness context')
    target_metal_species = _emc_context['target_metal_species']
    pressure = _emc_context['pressure']
    fix_indices = _emc_context['fix_indices']
    raw_score_cutoff = _emc_context['raw_score_cutoff']
    temperature_key = _emc_context['temperature_key']
    replica_key = _emc_context['replica_key']
    dir_name = _emc_context['work_dir']

    optimizer_class = _get_optimizer(optimizer)

    if len(calculator_list) != len(fmax_list):
        raise ValueError('calculator_list and fmax_list must have the same length')
    if run_md and md_type not in ('berendsen', 'langevin'):
        raise ValueError("md_type must be 'berendsen' or 'langevin'")
    if len(calculator_list) == 0:
        raise ValueError('calculator_list cannot be empty')
    T = atoms.info['key_value_pairs'][temperature_key]
    rep = atoms.info['key_value_pairs'][replica_key]
    atoms = rattle_close_atoms(atoms.copy(), min_distance=0.1)
    md_atoms = None
    if fix_indices is None:
        fix_indices = []
    else:
        atoms.set_constraint(FixAtoms(indices=fix_indices))
    mut_metal_indices = [a.index for a in atoms if (a.index not in fix_indices) 
                         and (a.symbol in target_metal_species)]
    vac_indices = [a.index for a in atoms if a.symbol == 'X']
    real_indices = [a.index for a in atoms if a.symbol != 'X']    
    tmp_atoms = atoms[real_indices].copy()
    if write_relax_traj:
        relax_traj = f'{dir_name}/relaxation_{rep}.traj'
    else:
        relax_traj = None

    for i, calc in enumerate(calculator_list):
        fmax = fmax_list[i]
        if i < len(calculator_list) - 1:
            tmp_atoms.calc = calc
            if fmax is not None:
                opt = optimizer_class(tmp_atoms, logfile=None, trajectory=relax_traj, append_trajectory=True)
                opt.run(fmax=fmax, steps=max_relax_steps)
            E0 = tmp_atoms.get_potential_energy()
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
                md_logfile = tag + f'_{rep}.log' if write_md_traj else None
                md_trajfile = tag + f'_{rep}.traj' if write_md_traj else None
                MaxwellBoltzmannDistribution(tmp_atoms, temperature_K=T, force_temp=True)
                Stationary(tmp_atoms)
                if pressure is None:
                    if md_type == 'berendsen':
                        dyn = NVTBerendsen(tmp_atoms, timestep=md_timestep, temperature_K=T,
                                           taut=md_taut, logfile=md_logfile,
                                           trajectory=None, append_trajectory=False)
                    elif md_type == 'langevin':
                        dyn = Langevin(tmp_atoms, timestep=md_timestep, temperature_K=T,
                                       friction=md_friction, logfile=md_logfile,
                                       trajectory=None, append_trajectory=False)
                else:
                    if md_type == 'berendsen':
                        dyn = NPTBerendsen(tmp_atoms, timestep=md_timestep, temperature_K=T,
                                           taut=md_taut, pressure_au=pressure,
                                           taup=md_taup, compressibility_au=compressibility,
                                           logfile=md_logfile, trajectory=None,
                                           append_trajectory=False)
                    elif md_type == 'langevin':
                        from ase.md.langevinbaoab import LangevinBAOAB
                        dyn = LangevinBAOAB(tmp_atoms, timestep=md_timestep, temperature_K=T,
                                            T_tau=md_taut, hydrostatic=True, externalstress=pressure,
                                            P_tau=md_taup, logfile=md_logfile,
                                            trajectory=None, append_trajectory=False)
                if write_md_traj:
                    traj = Trajectory(md_trajfile, 'a', tmp_atoms)
                    dyn.attach(traj.write, interval=md_write_interval)
                dyn.run(md_steps)
                if write_md_traj:
                    traj.close()
                if update_md_positions:
                    md_atoms = tmp_atoms.copy()
            penalize = False
            if fmax is not None:
                if (pressure is None) or not optimize_cell:
                    opt = optimizer_class(tmp_atoms, logfile=None, trajectory=relax_traj, append_trajectory=True)
                else:
                    ucf = FrechetCellFilter(tmp_atoms, scalar_pressure=pressure)
                    opt = optimizer_class(ucf, logfile=None, trajectory=relax_traj, append_trajectory=True)
                converged = opt.run(fmax=fmax, steps=max_relax_steps)
                if not converged:
                    penalize = True

    epot = tmp_atoms.get_potential_energy()
    if penalize:
        pseudo_atoms = copy.deepcopy(atoms)
        volume = atoms.get_volume()
        pseudo_atoms.info['key_value_pairs']['raw_score'] = raw_score_cutoff
    else:
        if (md_atoms is not None) and not update_relaxed_positions:
            pseudo_atoms = insert_vacancies(atoms, md_atoms, target_metal_species, neighbor_cutoff)
        elif update_relaxed_positions:
            pseudo_atoms = insert_vacancies(atoms, tmp_atoms, target_metal_species, neighbor_cutoff)
        else:
            pseudo_atoms = copy.deepcopy(atoms)
        volume = tmp_atoms.get_volume()
        pseudo_atoms.info['key_value_pairs']['raw_score'] = -epot

        if any(a for a in pseudo_atoms[mut_metal_indices] if a.symbol not in target_metal_species):
            parprint(f'Something wrong with replica {rep}, where metal indices are mutated into non-metals')
            pseudo_atoms.info['key_value_pairs']['raw_score'] = raw_score_cutoff

    pseudo_atoms.info['key_value_pairs']['volume'] = volume
    pseudo_atoms.info['key_value_pairs']['potential_energy'] = epot
    
    return pseudo_atoms


def calculate_surface_fitness(
        atoms,
        calculator_list,
        fmax_list,
        md_timestep,
        md_taut,
        md_steps,
        md_write_interval,
        optimizer='SQNM',
        run_md=False,
        md_type='berendsen',
        md_friction=None,
        max_relax_steps=300,
        write_relax_traj=False,
        write_md_traj=False,
        update_md_positions=True,
        update_relaxed_positions=False,
        neighbor_cutoff=2.5,
        _emc_context=None):

    """Calculate the fitness of a surface GC-EMC candidate.

    Parameters
    ----------
    atoms : ase.Atoms
        Surface structure to evaluate.
    calculator_list : list of ase.calculators.calculator.Calculator
        Calculators applied sequentially. The final calculator is used for
        molecular dynamics, the final relaxation and the reported energy.
    fmax_list : list of float or None
        Force-convergence thresholds corresponding to ``calculator_list``.
        A value of ``None`` skips relaxation with that calculator.
    md_timestep : float
        Molecular-dynamics time step in ASE time units.
    md_taut : float
        Temperature-coupling time constant in ASE time units.
    md_steps : int
        Number of molecular-dynamics steps.
    md_write_interval : int
        Interval between frames in the optional molecular-dynamics trajectory.
    optimizer : str, default 'SQNM'
        Structure optimizer. ``'SQNM'`` uses :class:`catemc.optimizers.SQNM`;
        other names are resolved from :mod:`ase.optimize`.
    run_md : bool, default False
        Run molecular dynamics before the final relaxation.
    md_type : {"berendsen", "langevin"}, default "berendsen"
        Molecular-dynamics thermostat.
    md_friction : float or None, default None
        Langevin friction coefficient. Used only when ``md_type="langevin"``.
    max_relax_steps : int, default 300
        Maximum number of optimization steps for each relaxation.
    write_relax_traj : bool, default False
        Write temporary relaxation trajectories.
    write_md_traj : bool, default False
        Write temporary molecular-dynamics trajectories and logs.
    update_md_positions : bool, default True
        Use the post-MD coordinates when constructing the returned candidate.
    update_relaxed_positions : bool, default False
        Use the fully relaxed coordinates in the returned candidate. Otherwise,
        relaxation is used only to evaluate its fitness.
    neighbor_cutoff : float, default 2.5
        Neighbor cutoff in angstrom used to map vacancies after atomic motion.
    _emc_context : dict or None, default None
        Internal GCEMC state supplied by the simulation driver. It contains the
        thermodynamic, structural and output settings that are intentionally not
        part of ``fitness_kwargs``.

    Returns
    -------
    ase.Atoms
        Evaluated candidate with its potential energy and raw score.
    """

    if _emc_context is None:
        raise ValueError('Missing GCEMC fitness context')
    target_metal_species = _emc_context['target_metal_species']
    target_metal_composition = _emc_context['target_metal_composition']
    allowed_adsorbate_species = _emc_context['allowed_adsorbate_species']
    element_column_map = _emc_context['element_column_map']
    element_eref_map = _emc_context['element_eref_map']
    element_y2x_map = _emc_context['element_y2x_map']
    adsorption_sites = _emc_context['adsorption_sites']
    slab_indices = _emc_context['slab_indices']
    fix_indices = _emc_context['fix_indices']
    temperature_dmu_map = _emc_context['temperature_dmu_map']
    adsorbate_surface_dmax = _emc_context['adsorbate_surface_dmax']
    raw_score_cutoff = _emc_context['raw_score_cutoff']
    temperature_key = _emc_context['temperature_key']
    replica_key = _emc_context['replica_key']
    ignore_sites = _emc_context['ignore_sites']
    dir_name = _emc_context['work_dir']

    optimizer_class = _get_optimizer(optimizer)

    if len(calculator_list) != len(fmax_list):
        raise ValueError('calculator_list and fmax_list must have the same length')
    if run_md and md_type not in ('berendsen', 'langevin'):
        raise ValueError("md_type must be 'berendsen' or 'langevin'")
    if len(calculator_list) == 0:
        raise ValueError('calculator_list cannot be empty')
    T = atoms.info['key_value_pairs'][temperature_key]
    rep = atoms.info['key_value_pairs'][replica_key]
    sas = adsorption_sites
    mu = atoms.info['data'].get('mu', None)
    atoms = rattle_close_atoms(atoms.copy(), min_distance=0.1)
    md_atoms = None
    if fix_indices is None:
        fix_indices = []
    else:
        atoms.set_constraint(FixAtoms(indices=fix_indices))
    mut_metal_indices = [a.index for a in atoms if (a.index not in fix_indices) 
                         and (a.symbol in target_metal_species)]
    vac_indices = [a.index for a in atoms if a.symbol == 'X']
    real_indices = [a.index for a in atoms if a.symbol != 'X']    
    tmp_atoms = atoms[real_indices].copy()
    tmp_slab_indices = [a.index for a in tmp_atoms if 
                        real_indices[a.index] in slab_indices]
    if write_relax_traj:
        relax_traj = f'{dir_name}/relaxation_{rep}.traj'
    else:
        relax_traj = None

    for i, calc in enumerate(calculator_list):
        fmax = fmax_list[i]
        if i < len(calculator_list) - 1:
            tmp_atoms.calc = calc
            if fmax is not None:
                opt = optimizer_class(tmp_atoms, logfile=None, trajectory=relax_traj, append_trajectory=True)
                opt.run(fmax=fmax, steps=max_relax_steps)
            E0 = tmp_atoms.get_potential_energy()
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
                md_logfile = tag + f'_{rep}.log' if write_md_traj else None
                md_trajfile = tag + f'_{rep}.traj' if write_md_traj else None
                MaxwellBoltzmannDistribution(tmp_atoms, temperature_K=T, force_temp=True)
                Stationary(tmp_atoms)
                if md_type == 'berendsen':
                    dyn = NVTBerendsen(tmp_atoms, timestep=md_timestep, temperature_K=T,
                                       taut=md_taut, logfile=md_logfile,
                                       trajectory=None, append_trajectory=False)
                elif md_type == 'langevin':
                    dyn = Langevin(tmp_atoms, timestep=md_timestep, temperature_K=T,
                                   friction=md_friction, logfile=md_logfile,
                                   trajectory=None, append_trajectory=False)
                if write_md_traj:
                    traj = Trajectory(md_trajfile, 'a', tmp_atoms)
                    dyn.attach(traj.write, interval=md_write_interval)
                dyn.run(md_steps)
                if write_md_traj:
                    traj.close()
                if update_md_positions:
                    md_atoms = tmp_atoms.copy()

            penalize = False
            if fmax is not None:
                opt = optimizer_class(tmp_atoms, logfile=None, trajectory=relax_traj, append_trajectory=True)
                converged = opt.run(fmax=fmax, steps=max_relax_steps)
                if not converged:
                    penalize = True
    if not penalize:
        if any(a for a in tmp_atoms if a.position[2] < atoms.positions[:,2].min()):
            penalize = True
        if any(a for a in tmp_atoms if (a.symbol in target_metal_species) and (
        adsorbate_surface_dmax is not None) and (
        a.position[2] > atoms.positions[:,2].max() + adsorbate_surface_dmax)):
            penalize = True
        try:
            nsas = SlabAdsorptionSites(tmp_atoms, surface='autoadsorbate',
                                       composition_effect=False,
                                       ignore_sites=ignore_sites)
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
            if md_atoms is not None:
                md_atoms = md_atoms[[a.index for a in md_atoms if
                                     real_indices[a.index] in slab_indices+adsorbate_indices]]
            atoms = atoms[[a.index for a in atoms if a.index in slab_indices+adsorbate_indices]]
            real_indices = [a.index for a in atoms if a.symbol != 'X']
            vac_indices = [a.index for a in atoms if a.symbol == 'X']
            if fix_indices is not None:
                atoms.set_constraint(FixAtoms(indices=fix_indices))
            tmp_atoms = new_atoms.copy()
            tmp_atoms.calc = calc
            if fmax is not None:
                opt = optimizer_class(tmp_atoms, logfile=None, trajectory=relax_traj, append_trajectory=True)
                converged = opt.run(fmax=fmax, steps=max_relax_steps)
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
        pseudo_atoms = copy.deepcopy(atoms)
        pseudo_atoms.info['key_value_pairs']['raw_score'] = raw_score_cutoff
        fragments = ([] if element_column_map is None else
                     [a.symbol for a in pseudo_atoms
                      if a.symbol in element_column_map])
    else:
        if (md_atoms is not None) and not update_relaxed_positions:
            pseudo_atoms = insert_vacancies(atoms, md_atoms, target_metal_species, neighbor_cutoff)
        elif update_relaxed_positions:
            pseudo_atoms = insert_vacancies(atoms, tmp_atoms, target_metal_species, neighbor_cutoff)
        else:
            pseudo_atoms = copy.deepcopy(atoms)
        nonmetal_species = [] if (element_column_map is None) else list(element_column_map.keys())
        num_dict = dict((m, len([a for a in tmp_atoms if a.symbol==m])) 
                        for m in target_metal_species + nonmetal_species)
        num_metal = len([a for a in tmp_atoms if a.symbol in target_metal_species])
        if mu is None:
            cond_arr = np.array([T])
        else:
            cond_arr = np.array([np.append(mu, T)])
        phi = compute_phi(num_dict, target_metal_species, target_metal_composition, temperature_dmu_map, cond_arr, 
                          element_column_map, element_eref_map, element_y2x_map, num_metal, epot, dimensionless=False)[0]
        pseudo_atoms.info['key_value_pairs']['raw_score'] = -phi
        fragments = adsorbates
        if element_column_map is not None:
            fragments += [a.symbol for a in tmp_atoms if (a.symbol in element_column_map.keys())
                          and (real_indices[a.index] not in adsorbate_indices)]                       
        if any(a for a in pseudo_atoms[mut_metal_indices] if a.symbol not in target_metal_species):
            parprint(f'Something wrong with replica {rep}, where metal indices are mutated into non-metals')
            pseudo_atoms.info['key_value_pairs']['raw_score'] = raw_score_cutoff
    pseudo_atoms.info['data']['fragments'] = fragments
    pseudo_atoms.info['key_value_pairs']['potential_energy'] = epot

    return pseudo_atoms


def insert_vacancies(atoms, relaxed_atoms, target_metal_species, neighbor_cutoff=2.5):
    """
    Vacancy insertion based on centroid update.

    Parameters
    ----------
    atoms: ase.Atoms
        Original structure containing 'X' vacancy atoms.

    relaxed_atoms: ase.Atoms
        Relaxed structure containing only the real atoms.

    target_metal_species: list of strs
        Metal species that can be considered as neighbors of vacancies.

    neighbor_cutoff: float
        Cutoff (Å) for NeighborList.

    Returns
    -------
    pseudo_atoms: ase.Atoms
        Copy of `atoms` with real atoms and cell from `tmp_atoms` and 
        inserted 'X' positions.

    """

    tmp_atoms = relaxed_atoms.copy()
    # Indices mapping between original atoms and relaxed atoms
    vac_indices = [i for i, a in enumerate(atoms) if a.symbol == 'X']
    real_indices = [i for i, a in enumerate(atoms) if a.symbol != 'X']
    mask = np.ones(len(atoms), dtype=bool)
    mask[vac_indices] = False
    new_to_orig = np.arange(len(atoms))[mask].tolist()
    orig_to_new = [None] * len(atoms)
    for new_j, orig_i in enumerate(new_to_orig):
        orig_to_new[orig_i] = new_j

    # neighborlist built on original full-atom topology (to find which atoms neighbor the vacancy)
    nl = NeighborList([neighbor_cutoff] * len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)

    # minimum allowed distance to any relaxed atom
    min_distance = 1.5
    eps = 1e-7

    # relaxed positions and cell for MIC unwrapping and vectorized distance
    R = tmp_atoms.get_positions()   # (N_real,3)
    cell = tmp_atoms.get_cell().copy()   # (3,3)
    inv_cell = np.linalg.inv(cell)
    orig_cell = atoms.get_cell().copy()
    orig_inv = np.linalg.inv(orig_cell)

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
    # R_frac = R @ inv_cell
    for vi in vac_indices:
        neigh_inds, _ = nl.get_neighbors(vi)
        # collect mapped relaxed neighbor positions
        neigh_pos = []
        for mi in neigh_inds:
            if atoms[mi].symbol in target_metal_species:
                mapped = orig_to_new[mi]
                if mapped is not None:
                    neigh_pos.append(R[mapped])
        if len(neigh_pos) == 0:
            # no metal neighbors -> keep the original vacancy location
            orig_frac = atoms.positions[vi] @ orig_inv
            vac_positions.append(orig_frac @ cell)
            continue
        neigh_pos = np.asarray(neigh_pos)   # (M,3)
        # Compute centroid using MIC relative to the vacancy position
        # Convert both neighbor positions and the vacancy position into fractional coords
        # using the relaxed cell, then wrap neighbor fractional differences
        # to [-0.5,0.5) relative to vacancy fractional coordinate, then convert back.
        vac_frac = (atoms.positions[vi] @ inv_cell)
        neigh_frac = neigh_pos @ inv_cell           
        # fractional difference from vacancy to each neighbor, wrapped to [-0.5,0.5)
        delta_frac = neigh_frac - vac_frac           
        delta_frac -= np.round(delta_frac)           
        neigh_images_frac = vac_frac + delta_frac    
        neigh_images_cart = neigh_images_frac @ cell 
        centroid = np.mean(neigh_images_cart, axis=0)
        # check MIC min distance from centroid to relaxed atoms
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
        # if candidate is still too close, fallback to original vacancy position
        min_d2, _, _ = min_image_min_dist_and_idx(candidate)
        if min_d2 < min_distance - eps:
            orig_frac = atoms.positions[vi] @ orig_inv
            candidate = orig_frac @ cell
        vac_positions.append(candidate)

    # create pseudo_atoms: copy original, replace real atom positions by relaxed positions,
    # and insert computed vacancy 'X' positions
    pseudo_atoms = copy.deepcopy(atoms)
    if len(real_indices) != 0:
        pseudo_atoms.positions[real_indices] = R
    if vac_indices:
        pseudo_atoms.positions[vac_indices] = np.vstack(vac_positions)
    pseudo_atoms.cell = cell

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
