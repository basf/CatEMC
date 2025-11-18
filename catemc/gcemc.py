#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Algorithm for grand canonical evolutionary Monte Carlo (GC-EMC) run"""
from .emc_operators import (
        EMCCutSpliceCrossover,
        EMCCatalystAdsorbateCrossover,
        EMCInterSwap,
        EMCIntraSwap,
        EMCMutation)
from .emc import EMC
from .gc_utils import compute_phi
from acat.adsorption_sites import SlabAdsorptionSites
from acat.ga.adsorbate_operators import (
        AddAdsorbate, 
        RemoveAdsorbate,                                 
        MoveAdsorbate, 
        ReplaceAdsorbate,
        ReplaceAdsorbateSpecies)
from ase.io import read, write
from ase.db import connect
from ase import units
from ase.io.trajectory import Trajectory
from ase.ga.data import DataConnection, PrepareDB
from ase.parallel import parprint
from collections import defaultdict
from itertools import combinations
import numpy as np
import warnings
import random
import shutil
import pickle
import math
import time
import os


class GCEMC(EMC):
    """
    Main algorithm for grand canonical evolutionary Monte Carlo (GC-EMC).

    Parameters
    ----------
    name: str
        Name of the target system.
    
    calculation_dir: str
        Path of the calculation directory.

    calculation_script: str
        The name of the Python script for calculating the fitness of each
        configuration. The script must be inside calculation_dir.
    
    input_pop: list of ase.Atoms objects
        Input population for the EMC run.
    
    fix_indices: list of ints, default None
        A list of atom indices that should be immutable and fixed in positions
        throughout the EMC run. Nothing is fixed by default.

    trim_generation: int, default 100
        To trim database (with backup) every N steps. The database file would be 
        too large and adding new data would be too slow without trimming.

    md_generation: int, default 10
        To perform molecular dynamics every N steps.

    mu_walk_generation: int, default 1
        To perform mu-walk operation every N steps.

    exchange_generation: int, defatul 1
        To perform replica exchange every N steps.

    bias_starting_step: int, default None
        From which step should a bias weight be considered in the mu-walk
        operations to uniformly sample the chemical potential space. The bias
        weight used here is the dimensionless free energy estimated by MBAR.
        Chemical potential space is not sampled uniformly if this is not set.

    mbar_starting_step: int, default 100
        From which step should MBAR starts sampling data for estimating free
        energy differences to uniformly sample the chemical potential space.

    mbar_sample_stride: int, default 10
        The number of steps that separates two samples at each state for MBAR 
        free energy difference calculations to uniformly sample the chemical 
        potential space.

    crossover_rate: float, default 0.
        The rate of performing crossover operations.

    interswap_rate: float, default 0.
        The rate of performing interswap operations.

    slab_intraswap_rate: float, default 0.
        The rate of performing intraswap operations in the catalyst slab.

    slab_mutation_rate: float, default 0.
        The rate of performing intraswap operations in the catalyst slab.

    adsorbate_insertion_rate: float, default 0.
        The rate of adding an adsorbate fragment to the catalyst surface.

    adsorbate_removal_rate: float, default 0.
        The rate of removing an adsorbate fragment from the catalyst surface.

    adsorbate_displacement_rate: float, default 0.
        The rate of moving an adsorbate fragment from one adsorption site to
        another on the catalyst surface.

    adsorbate_mutation_rate: float, default 0.
        The rate of mutating the type of an adsorbate fragment to another type.

    adsorbate_surface_dmax: float, default 2.5
        The distance to the catalyst surface (in Å) that a fragment should still 
        be considered as an adsorbate. Fragments that are further away more than
        this value from the catalyst surface will be removed from the simulation 
        cell automatically.

    adsorbate_adsorbate_dmin: float, default 2.
        The minimum adsorbate-adsorbate distance (in Å) that allowed during the
        EMC run. Effectively setting a maximum coverage while avoid overlapping.

    max_adsorbate_species: int, default None
        The maximum number of unique adsorbate fragment types allowed in a single
        simulation cell during the EMC run. No such restriction if this is not set.

    max_adsorbate_coverage: float, default 1.
        The maximum adsorbate coverage (in ML) allowed in a single simulation cell
        during the EMC run.

    metal_element_pool: list of strs, default None
        The list of target metal elements (including 'X' which represents vacancy) 
        that should be mutable during the EMC run.

    nonmetal_element_pool: list of strs, default None
        The list of target non-metal elements (including 'X' which represents 
        vacancy) that should be mutable during the EMC run.

    bulk_metal_composition: dict of (str, float) pairs, default None
        The reference metal composition of the bulk catalyst system.

    adsorbate_species: list of strs, default None
        The list of adsorbate fragment types used to perform adsorbate operations.
        The internal partition function values for such species must be provided 
        in temperature_species_internal_map.

    other_adsorbate_species: list of strs, default None
        The list of adsorbate fragment types that are not involved in adsorbate
        operations but is still allowed during the EMC simulation. The internal
        partition function parameters for such species must be provided in
        temperature_species_internal_map.

    mutable_metal_indices: list of ints, default None
        The list of metal atom (or 'X' which represents vacancy) indices that are 
        allowed to mutate during the EMC run. Default None means all metal indices 
        are mutable.

    mutable_nonmetal_indices: list of ints, default None
        The list of non-metal (or 'X' which represents vacancy) indices that are
        allowed to mutate during the EMC run. Default None means all non-metal 
        indices are mutable.

    slab_indices: list of ints, default None
        The list of atom indices that corresponds to the catalyst slab. Note that 
        these atom indices will be preserved throughout the EMC run.

    volume: float, default None
        The accessible volume for the simlated grand canonical system. Note that the
        metal-atom-occupied space should be subtracted from the volume calculation.

    reaction_conditions: numpy.ndarray, default None
        The chemical potential and temperature array that represents the reaction
        condition grid to explore. Each row represents a unique reaction condition,
        and temperatures must be included in the last column.

    element_column_map: dict of (str, int) pairs, default None
        A dictionary that maps each non-metal element to the column index that
        corresponds to the chemical potential of the element.

    element_eref_map: dict of (str, float) pairs, default None
        A dictionary that maps each non-metal element to the reference total energy
        of the element. This is only useful if the provided reaction_conditions
        include relative chemical potentials. No need to set this if only absolute 
        chemical potentials are included, since zero reference energies are set by 
        default. Only needed if the free energy bias weight is applied.

    element_y2x_map: dict of (str, float) pairs, default None
        A dictionary that maps each non-metal element to the reference bulk relative 
        stoichiometry to the metal in a M_x{NM_y} compound. E.g. for a MnFeCO3 bulk
        compound should be set as {'C': 1/2, 'O': 3/2}. No need to set this if the
        system only has metal atoms, since y2x values are set to zero by default.
        Only needed if the free energy bias weight is applied.

    temperature_dmu_map: dict of (float, float) pairs, default None
        A dictionary that maps each temperature to the estimated the chemical
        potential difference between all metal pairs (e.g. obtained from bulk GE-EMC 
        simulations). Note that the keys must be in the form of 'A2B' which represents
        mu_B - mu_A, must consider all possible metal pairs, and the metal element 
        order must follow the order in the metal_element_pool. Only needed if the free 
        energy bias weight is applied. Default is None, which is only suitable for 
        surface EMC simulations with fixed surface metal composition.

    temperature_species_internal_map: dict of (float, dict) pairs, default None
       A dictionary that maps the temperature to the dictionary of adsorbate fragments'
       internal partition function values. 

    adsorption_sites: acat.SlabAdsorptionSites object, default None
       Fallback option when the AutoAdsorbate adsorption sites identification failed
       in the first step. In most cases not needed.

    ignore_sites: str or list of strs, default None
        The catalyst surface adsorption sites that should not be considered for 
        adsorbate insertion. Common practice is to ignore 'bridge' sites to accelerate 
        the EMC run.

    raw_score_cutoff: float, default -10000.
        The fitness score for candidates that are should not be considered in the 
        sampling. This could be structures that are unphysical / failed to relax by 
        the calculator.

    temperature_key: str, default 'temperature'
        Keyword for the temperature stored in each configuration.

    replica_key: str, default 'replica'
        Keyword for the replica number stored in each configuration.

    acceptance_key: str, default 'accepted'
        Keyword for indicating being accepted or not in each configuration

    gpu_devices: list of ints, default None
        The list of GPU devices to use. Default is to use a single GPU/CPU device.

    procs_per_device: int, default 1
        The number of processes to run in parallel on each GPU/CPU device. Defaul is 
        to run in serial.

    batching: bool, default False
        Whether to run batch relaxation and MD using torch-sim. Default is False.

    """

    def __init__(self, name,
            calculation_dir,
            calculation_script,
            input_pop,
            fix_indices=None,
            trim_generation=100,
            md_generation=10,
            mu_walk_generation=1,
            exchange_generation=1,
            bias_starting_step=None,
            mbar_starting_step=100,
            mbar_sample_stride=10,
            crossover_rate=0.,
            interswap_rate=0.,
            slab_intraswap_rate=0.,
            slab_mutation_rate=0.,
            adsorbate_insertion_rate=0.,
            adsorbate_removal_rate=0.,
            adsorbate_displacement_rate=0.,
            adsorbate_mutation_rate=0.,
            adsorbate_surface_dmax=2.5,
            adsorbate_adsorbate_dmin=2.,
            max_adsorbate_species=None,
            max_adsorbate_coverage=1.,
            metal_element_pool=None,
            nonmetal_element_pool=None,
            bulk_metal_composition=None,
            adsorbate_species=None,
            other_adsorbate_species=None,
            mutable_metal_indices=None,
            mutable_nonmetal_indices=None,
            slab_indices=None,
            volume=None,
            reaction_conditions=None,
            element_column_map=None,
            element_eref_map=None,
            element_y2x_map=None,
            temperature_dmu_map=None,
            temperature_species_internal_map=None,
            adsorption_sites=None,
            ignore_sites=None,
            raw_score_cutoff=-10000.,
            temperature_key='temperature',
            replica_key='replica',
            acceptance_key='accepted',
            gpu_devices=None,
            procs_per_device=1,
            batching=False,
            verbose=True):

        super().__init__(
                name=name, 
                calculation_dir=calculation_dir,
                calculation_script=calculation_script,
                input_pop=input_pop,
                fix_indices=fix_indices,
                trim_generation=trim_generation,
                md_generation=md_generation,
                exchange_generation=exchange_generation,
                metal_element_pool=metal_element_pool,
                nonmetal_element_pool=nonmetal_element_pool,
                mutable_metal_indices=mutable_metal_indices,
                mutable_nonmetal_indices=mutable_nonmetal_indices,
                raw_score_cutoff=raw_score_cutoff,
                temperature_key=temperature_key,
                replica_key=replica_key,
                acceptance_key=acceptance_key,
                gpu_devices=gpu_devices,
                procs_per_device=procs_per_device,
                batching=batching,
                verbose=verbose)

        self.mu_walk_generation = mu_walk_generation
        self.bias_starting_step = bias_starting_step
        self.mbar_starting_step = mbar_starting_step
        self.mbar_sample_stride = mbar_sample_stride
        self.crossover_rate = crossover_rate
        self.interswap_rate = interswap_rate
        self.slab_intraswap_rate = slab_intraswap_rate
        self.slab_mutation_rate = slab_mutation_rate
        self.adsorbate_insertion_rate = adsorbate_insertion_rate
        self.adsorbate_removal_rate = adsorbate_removal_rate
        self.adsorbate_displacement_rate = adsorbate_displacement_rate
        self.adsorbate_mutation_rate = adsorbate_mutation_rate
        self.adsorbate_surface_dmax = adsorbate_surface_dmax
        self.adsorbate_adsorbate_dmin = adsorbate_adsorbate_dmin
        self.max_adsorbate_species = max_adsorbate_species
        self.max_adsorbate_coverage = max_adsorbate_coverage
        self.bulk_metal_composition = bulk_metal_composition
        self.adsorbate_species = adsorbate_species
        self.other_adsorbate_species = other_adsorbate_species
        self.slab_indices = slab_indices
        self.volume = volume
        self.reaction_conditions = reaction_conditions
        self.element_column_map = element_column_map
        self.element_eref_map = element_eref_map
        self.element_y2x_map = element_y2x_map
        self.tdmu_map = temperature_dmu_map
        self.tsi_map = temperature_species_internal_map
        self.adsorption_sites = adsorption_sites
        self.ignore_sites = ignore_sites

    def calculate_free_energy_differences(self, starting_step, ending_step, sample_stride, tol=1e-3):

        from pymbar import MBAR
        all_species = self.metal_element_pool + list(self.element_column_map.keys())
        sim_temperatures = sorted({c.info['key_value_pairs'][self.temperature_key] for c in self.pop.pop})
        def create_energy_list():
            return [None for _ in range(ending_step)]
        energy_data = {T: defaultdict(create_energy_list) for T in sim_temperatures}
        n = 0
        while os.path.isfile(f'{self.calculation_dir}/{self.name}_bak{n}.db'):
            n += 1
        for i in range(n):
            if (i+1) * self.trim_generation <= starting_step:
                continue
            db = connect(f'{self.calculation_dir}/{self.name}_bak{i}.db')
            for i, row in enumerate(db.select('relaxed=1')):
                if row.emc_step >= starting_step:
                    T = row.temperature
                    if row.key_value_pairs[self.acceptance_key] == 1:
                        if row.potential_energy > 0.:
                            continue
                        if row.emc_step > ending_step:
                            continue
                        if row.raw_score <= self.raw_score_cutoff:
                            continue
                        atoms = row.toatoms()
                        atoms.info = {'key_value_pairs': {self.temperature_key: T, 
                                                          self.replica_key: row.key_value_pairs[self.replica_key]},
                                      'data': {}}
                        atoms.info['key_value_pairs']['potential_energy'] = row.potential_energy
                        atoms.info['data']['mu'] = row.data.mu
                        if 'mu_window' in row.data:
                            atoms.info['data']['mu_window'] = row.data.mu_window
                        energy_data[T][row.key_value_pairs[self.replica_key]][row.emc_step-1] = atoms
 
        # Assign the previous energy if energy is None at a step
        for T, dd in energy_data.items():
            for rep in dd.keys():
                U = None
                dd[rep] = [U:=E if E is not None else U for E in dd[rep]]

        grid = self.reaction_conditions.copy()
        # Build MBAR inputs
        K = len(grid) # No. of states
        samples_per_state = np.zeros(K, dtype=int)
        # Fill energy matrix
        u_kn = []
        sample_state = []
        sample_steps = []
        for k, T in enumerate(sim_temperatures):
            for rep, lol in energy_data[T].items():
                for n_minus_1, atoms in enumerate(lol):
                    if atoms is None:
                        continue
                    epot = atoms.info['key_value_pairs']['potential_energy']
                    num_metal = len([a for a in atoms if a.symbol in self.metal_element_pool])
                    num_dict = dict((m, len([a for a in atoms if a.symbol==m])) for m in all_species)
                    phi = compute_phi(
                            num_dict, self.metal_element_pool, self.bulk_metal_composition, 
                            self.tdmu_map, grid, self.element_column_map, self.element_eref_map, 
                            self.element_y2x_map, num_metal, epot, dimensionless=True)
                    cond = np.append(atoms.info['data']['mu'], T)
                    grid_id = np.where(np.all(np.abs(grid - cond) < tol, axis=1))[0]
                    state_idx = int(grid_id[0])
                    samples_per_state[state_idx] += 1
                    sample_state.append(state_idx)
                    sample_steps.append(n_minus_1)
                    u_kn.append(phi.reshape(-1,1))
 
        u_kn = np.hstack(u_kn)  # Shape (K, N_total)
 
        # Build numpy array of which state produced each sample (global column index)
        sample_state = np.asarray(sample_state, dtype=int) # Shape (N_total,)
        sample_steps = np.asarray(sample_steps, dtype=int) # Shape (N_total,)
        N_total = sample_state.size
        # Per-state subsampling using simulation-step bin width = stride:
        # For each state k: find indices idx_k (global indices of samples assigned to that state),
        # get their simulation steps steps_k = sample_steps[idx_k],
        # bin as bins = steps_k // stride_int and keep the FIRST sample in each unique bin.
        stride_int = int(sample_stride)
        kept_mask = np.zeros(N_total, dtype=bool)
        for k_state in range(K):
            idx_k = np.where(sample_state == k_state)[0]
            if idx_k.size == 0:
                continue
            steps_k = sample_steps[idx_k]
            # compute integer bins by simulation step
            bins = steps_k // stride_int
            # unique bins and the first occurrence index within steps_k
            #unique_bins, first_pos = np.unique(bins, return_index=True)
            #subs_global = idx_k[first_pos]  # global indices to keep for this state
            unique_bins, last_pos_rev = np.unique(bins[::-1], return_index=True)
            last_pos = bins.size - 1 - last_pos_rev
            subs_global = idx_k[last_pos]
            kept_mask[subs_global] = True
 
        kept_indices = np.nonzero(kept_mask)[0]
 
        # make subsampled phi_kn and per-state counts for MBAR
        u_kn_sub = u_kn[:, kept_indices]
        samples_per_state_sub = np.array([np.sum(sample_state[kept_indices] == k) for k in range(K)], dtype=int)
        parprint(f'New per-state sample counts (subsampled): {samples_per_state_sub}')
 
        mbar = MBAR(u_kn_sub, samples_per_state_sub, relative_tolerance=1e-7,
                    solver_protocol='robust', maximum_iterations=30000)
        results = mbar.compute_free_energy_differences()
        df = results['Delta_f']
        np.save(f'{self.calculation_dir}/{self.name}_free_energy_differences.npy', df)

    def run(self, starting_step=1, total_steps=10000, restart_with_trim=False):

        """                                    
        Parameters
        ----------
        starting_step: int, default 1
            (Re)starting from a certain EMC step.
                                               
        total_steps: int, default 10000
            The total number of EMC steps that the algorithm should run.

        restart_with_trim: bool, default False
            Whether to restart while trimming all data generated after
            the starting_step that has been generated before.

        """
             
        restart = True
        self.pop_size = len(self.input_pop)
        # Connect to the ase db
        asedb_path = self.calculation_dir + '/{}.db'.format(self.name)
        if not os.path.isfile(asedb_path):
            restart = False
            asedb = PrepareDB(asedb_path, population_size=self.pop_size)
            for atoms in self.input_pop:
                if 'key_value_pairs' not in atoms.info:
                    atoms.info['key_value_pairs'] = {}
                if 'data' not in atoms.info:
                    atoms.info['data'] = {}
                kwargs = {self.temperature_key: atoms.info['key_value_pairs'][self.temperature_key], 
                          self.replica_key: atoms.info['key_value_pairs'][self.replica_key],
                          self.acceptance_key: 0}
                asedb.add_unrelaxed_candidate(atoms, 
                        emc_step=atoms.info['key_value_pairs']['emc_step'],
                        data=atoms.info['data'],
                        **kwargs)
        if restart_with_trim:
            self.connect_data_with_trim(starting_step=starting_step)
        else:
            self.dc = DataConnection(asedb_path)
        if not os.path.isdir(f'{self.calculation_dir}/tmp_{self.name}'):
            os.mkdir(f'{self.calculation_dir}/tmp_{self.name}')
        if self.slab_indices is None:
            slab_indices_str = 'None'
        else:
            slab_indices_str = ','.join(map(str, self.slab_indices)) 
        if not restart:
            unrelaxed_candidates = self.dc.get_all_unrelaxed_candidates()
            relaxed_candidates = self.get_relaxed_candidates(
                    unrelaxed_candidates, run_md=True, add_args=[slab_indices_str], batching=self.batching)
            for atoms in relaxed_candidates:
                rep = atoms.info['key_value_pairs'][self.replica_key]
                atoms.info['key_value_pairs'][self.acceptance_key] = 1
                try:
                    sas = SlabAdsorptionSites(atoms, surface='autoadsorbate', 
                                              composition_effect=False,
                                              ignore_sites=self.ignore_sites)
                    with open(f'{self.calculation_dir}/tmp_{self.name}/sites_{rep}.pkl', 'wb') as handle:
                        pickle.dump(sas, handle, protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    with open(f'{self.calculation_dir}/tmp_{self.name}/sites_{rep}.pkl', 'wb') as handle:
                        pickle.dump(self.adsorption_sites, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            relaxed_candidates = []
        rates_sum = self.slab_intraswap_rate + self.slab_mutation_rate + self.crossover_rate + self.interswap_rate +\
                    self.adsorbate_insertion_rate + self.adsorbate_removal_rate + self.adsorbate_displacement_rate +\
                    self.adsorbate_mutation_rate
        if rates_sum != 1:
            self.slab_intraswap_rate /= rates_sum
            self.slab_mutation_rate /= rates_sum
            self.crossover_rate /= rates_sum
            self.interswap_rate /= rates_sum
            self.adsorbate_insertion_rate /= rates_sum
            self.adsorbate_removal_rate /= rates_sum
            self.adsorbate_displacement_rate /= rates_sum
            self.adsorbate_mutation_rate /= rates_sum

        self.ntask = 1
        self.pop = self.get_pop()
        if self.trim_generation == -1:
            parprint('Restarting from the final population of the previously-converged EMC run. ' + 
                     'All other previously-stored structures are discarded from the ASE database.')
            self.trim_asedb()
            self.pop = self.get_pop() 
        # Update the population to allow new candidates to enter
        self.dc.add_more_relaxed_candidates(relaxed_candidates)
        self.pop.update()
        if not restart:
            for atoms in self.pop.pop:
                #atoms = atoms[[a.index for a in atoms if a.symbol != 'X']]
                rep = atoms.info['key_value_pairs'][self.replica_key]
                traj = Trajectory(f'{self.calculation_dir}/tmp_{self.name}/emc_{rep}.traj', 'a')
                traj.write(atoms)
        if restart_with_trim:
            for atoms in self.pop.pop:
                rep = atoms.info['key_value_pairs'][self.replica_key]
                traj = read(f'{self.calculation_dir}/tmp_{self.name}/emc_{rep}.traj', index=f':{starting_step}')
                write(f'{self.calculation_dir}/tmp_{self.name}/emc_{rep}.traj', traj)

        start_time = time.perf_counter()
        # Below is the iterative part of the algorithm
        gen_num = starting_step 
        gi = 0
        while gen_num + gi <= total_steps:

            if (gen_num + gi) % self.trim_generation == 0:
                self.trim_asedb()
                self.pop = self.get_pop()
                self.pop.current_fitness = self.pop.__get_fitness__(self.pop.pop)
                parprint('ASE database trimed at generation {0}. Restart from current population.'.format(gen_num + gi))
            parprint('Creating and evaluating generation {0}'.format(gen_num + gi))
                
            if (self.bias_starting_step is not None) and (gen_num + gi >= self.bias_starting_step):
                df_path = f'{self.calculation_dir}/{self.name}_free_energy_differences.npy'
                if not os.path.isfile(df_path):
                    self.calculate_free_energy_differences(
                            starting_step=self.mbar_starting_step, 
                            ending_step=gen_num+gi, 
                            sample_stride=self.mbar_sample_stride)
            
            if (gen_num + gi) % self.md_generation == 0:
                unrelaxed_candidates = self.pop.pop
                relaxed_candidates = self.get_relaxed_candidates(
                        unrelaxed_candidates, run_md=True, add_args=[slab_indices_str], batching=self.batching)
                relaxed_candidates.sort(key=lambda c: c.info['key_value_pairs'][self.replica_key])
                for atoms in relaxed_candidates:
                    rep = atoms.info['key_value_pairs'][self.replica_key]
                    atoms.info['key_value_pairs'][self.acceptance_key] = 1
                    try:
                        sas = SlabAdsorptionSites(atoms, surface='autoadsorbate', 
                                              composition_effect=False,
                                              ignore_sites=self.ignore_sites)
                        with open(f'{self.calculation_dir}/tmp_{self.name}/sites_{rep}.pkl', 'wb') as handle:
                            pickle.dump(sas, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        pass
                # Update the population to allow new candidates to enter
                self.dc.add_more_relaxed_candidates(relaxed_candidates) 
                self.pop.update()
           
            n_crossover_pairs = round((self.pop_size * self.crossover_rate) / 2)
            n_interswap_pairs = round((self.pop_size * self.interswap_rate) / 2)
            all_parents = self.pop.get_all_parents(n_crossover_pairs, n_interswap_pairs,
                                                   interswap_within_temperature=True)
            
            for parents in all_parents:
                if len(parents) == 2:
                    if parents[0].info['data']['for_operation'] == 'crossover':
                        operator = EMCCatalystAdsorbateCrossover(catalyst_indices=self.slab_indices)
                    elif parents[0].info['data']['for_operation'] == 'interswap':
                        operator = EMCInterSwap(allowed_indices=self.mutable_metal_indices, 
                                                element_pools=self.metal_element_pool)
                    offsprings, desc = operator.get_new_individual(parents)
                    for idx in range(2):
                        rep = parents[idx].info['key_value_pairs'][self.replica_key]
                        if os.path.exists(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{rep}.traj'):
                            continue
                        if offsprings[idx] == parents[idx]:
                            warnings.warn(f'Offspring identical to parent. Something with replica {rep} went wrong')
                        offsprings[idx].info['key_value_pairs'][self.temperature_key] = parents[idx].info['key_value_pairs'][self.temperature_key]
                        offsprings[idx].info['key_value_pairs'][self.replica_key] = rep
                        offsprings[idx].info['data']['mu'] = parents[idx].info['data']['mu']
                        if 'mu_window' in parents[idx].info['data']:
                            offsprings[idx].info['data']['mu_window'] = parents[idx].info['data']['mu_window']
                        write(f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj', offsprings[idx])
            
                elif len(parents) == 1:
                    rep = parents[0].info['key_value_pairs'][self.replica_key]
                    if os.path.exists(f'{self.calculation_dir}/tmp_{self.name}/relaxed_atoms_{rep}.traj'):
                        continue
                    T = parents[0].info['key_value_pairs'][self.temperature_key]
                    species_probabilities = None
                    adsorbate_species = self.adsorbate_species.copy()
                    if self.other_adsorbate_species is not None:
                        species_probabilities = {k: 1. for k in adsorbate_species}
                        for k in self.other_adsorbate_species:
                            species_probabilities[k] = 0.
                        adsorbate_species += self.other_adsorbate_species

                    site_path = f'{self.calculation_dir}/tmp_{self.name}/sites_{rep}.pkl'
                    if (self.mutable_nonmetal_indices is None) or (self.nonmetal_element_pool is None):
                        allowed_indices = self.mutable_metal_indices
                        element_pools = self.metal_element_pool
                    else:
                        allowed_indices = [self.mutable_metal_indices, self.mutable_nonmetal_indices]
                        element_pools = [self.metal_element_pool, self.nonmetal_element_pool]
                    operators = [EMCIntraSwap(allowed_indices=allowed_indices, 
                                              element_pools=element_pools), 
                                 EMCMutation(allowed_indices=allowed_indices, 
                                             element_pools=element_pools,
                                             element_probabilities=self.bulk_metal_composition)]
                    weights = [self.slab_intraswap_rate, self.slab_mutation_rate]
                    if self.adsorbate_species is not None:
                        operators += \
                                [AddAdsorbate(adsorbate_species=adsorbate_species, 
                                              species_probabilities=species_probabilities,
                                              max_species=self.max_adsorbate_species, 
                                              max_coverage=self.max_adsorbate_coverage, 
                                              adsorption_sites=site_path, 
                                              min_adsorbate_distance=self.adsorbate_adsorbate_dmin, 
                                              catalyst_indices=self.slab_indices, 
                                              dmax=self.adsorbate_surface_dmax),
                                 RemoveAdsorbate(adsorbate_species=adsorbate_species, 
                                                 adsorption_sites=site_path, 
                                                 catalyst_indices=self.slab_indices, 
                                                 dmax=self.adsorbate_surface_dmax),
                                 MoveAdsorbate(adsorbate_species=adsorbate_species, 
                                               adsorption_sites=site_path, 
                                               catalyst_indices=self.slab_indices, 
                                               min_adsorbate_distance=self.adsorbate_adsorbate_dmin, 
                                               dmax=self.adsorbate_surface_dmax),
                                 ReplaceAdsorbate(adsorbate_species=adsorbate_species, 
                                                  species_probabilities=species_probabilities, 
                                                  max_species=self.max_adsorbate_species, 
                                                  adsorption_sites=site_path, 
                                                  catalyst_indices=self.slab_indices, 
                                                  min_adsorbate_distance=self.adsorbate_adsorbate_dmin, 
                                                  dmax=self.adsorbate_surface_dmax),
                                 ReplaceAdsorbateSpecies(adsorbate_species=adsorbate_species,
                                                         species_probabilities=species_probabilities, 
                                                         adsorption_sites=site_path, 
                                                         catalyst_indices=self.slab_indices, 
                                                         dmax=self.adsorbate_surface_dmax)]
                    weights += [self.adsorbate_insertion_rate, self.adsorbate_removal_rate, self.adsorbate_displacement_rate, 
                                self.adsorbate_mutation_rate*2/3, self.adsorbate_mutation_rate*1/3]
                    operator = random.choices(population=operators, weights=weights)[0]
                    offspring, desc = operator.get_new_individual(parents)
                    if offspring == parents[0]:
                        warnings.warn(f'Offspring identical to parent. Something with replica {rep} went wrong')
                    offspring.info['key_value_pairs'][self.temperature_key] = parents[0].info['key_value_pairs'][self.temperature_key]
                    offspring.info['key_value_pairs'][self.replica_key] = rep
                    offspring.info['data']['mu'] = parents[0].info['data']['mu']
                    if 'mu_window' in parents[0].info['data']:
                        offspring.info['data']['mu_window'] = parents[0].info['data']['mu_window']
                    write(f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj', offspring)
            
            replicas = sorted([c.info['key_value_pairs'][self.replica_key] for lst in all_parents for c in lst])
            unrelaxed_offsprings = [read(f'{self.calculation_dir}/tmp_{self.name}/atoms_{rep}.traj') for rep in replicas]
            relaxed_offsprings = self.get_relaxed_candidates(
                    unrelaxed_offsprings, run_md=False, add_args=[slab_indices_str], batching=self.batching)
            relaxed_candidates = []
            for parents in all_parents:
                if len(parents) == 2:
                    rep1 = parents[0].info['key_value_pairs'][self.replica_key]
                    rep2 = parents[1].info['key_value_pairs'][self.replica_key]
                    relaxed_offspring1 = relaxed_offsprings[replicas.index(rep1)]
                    relaxed_offspring2 = relaxed_offsprings[replicas.index(rep2)] 
                    relaxed_offspring1.info['data']['operation'] = parents[0].info['data']['for_operation']
                    relaxed_offspring2.info['data']['operation'] = parents[1].info['data']['for_operation']
                    relaxed_offspring1.info['key_value_pairs']['emc_step'] = gen_num + gi
                    relaxed_offspring2.info['key_value_pairs']['emc_step'] = gen_num + gi
                    relaxed_candidates += self.pop.evaluate_offsprings(
                            parents, [relaxed_offspring1, relaxed_offspring2], prefactor='gc', volume=self.volume,
                            nonmetal_species=self.adsorbate_species, temperature_species_internal_map=self.tsi_map,
                            no_internal_indices=self.mutable_nonmetal_indices, mutable_indices=self.mutable_metal_indices, 
                            raw_score_cutoff=self.raw_score_cutoff)
                elif len(parents) == 1:
                    rep = parents[0].info['key_value_pairs'][self.replica_key]
                    relaxed_offspring = relaxed_offsprings[replicas.index(rep)]
                    relaxed_offspring.info['data']['operation'] = parents[0].info['data']['for_operation']
                    relaxed_offspring.info['key_value_pairs']['emc_step'] = gen_num + gi
                    relaxed_candidates += self.pop.evaluate_offsprings(
                            parents, [relaxed_offspring], prefactor='gc', volume=self.volume,
                            nonmetal_species=self.adsorbate_species, 
                            temperature_species_internal_map=self.tsi_map,
                            no_internal_indices=self.mutable_nonmetal_indices, 
                            metal_species_probabilities=self.bulk_metal_composition,
                            mutable_indices=self.mutable_metal_indices,
                            raw_score_cutoff=self.raw_score_cutoff)
            
            # Update the population to allow new candidates to enter
            self.dc.add_more_relaxed_candidates(relaxed_candidates)
            self.pop.update()
            for atoms in relaxed_candidates:
                if atoms.info['key_value_pairs'][self.acceptance_key] == 1:
                    rep = atoms.info['key_value_pairs'][self.replica_key]
                    try:
                        sas = SlabAdsorptionSites(atoms, surface='autoadsorbate', 
                                              composition_effect=False,
                                              ignore_sites=self.ignore_sites)
                        with open(f'{self.calculation_dir}/tmp_{self.name}/sites_{rep}.pkl', 'wb') as handle:
                            pickle.dump(sas, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        pass
            if (self.mu_walk_generation is not None) and (self.mu_walk_generation > 0):
                if (gen_num + gi) % self.mu_walk_generation == 0:
                    if gen_num + gi >= self.bias_starting_step:
                        df_path = f'{self.calculation_dir}/{self.name}_free_energy_differences.npy'
                        df = np.load(df_path)
                    else:
                        df = None
                    self.pop.mu_walk(element_column_map=self.element_column_map,
                                     multiple_try=True,
                                     reaction_conditions=self.reaction_conditions,
                                     free_energy_differences=df)
            if (self.exchange_generation is not None) and (self.exchange_generation > 0):
                if (gen_num + gi) % self.exchange_generation == 0:
                    self.pop.exchange(
                            exchange_within_temperature=True, prefactor='gc', 
                            element_column_map=self.element_column_map,
                            temperature_species_internal_map=self.tsi_map,
                            no_internal_indices=self.mutable_nonmetal_indices)
            
            for atoms in self.pop.pop:
                #atoms = atoms[[a.index for a in atoms if a.symbol != 'X']]
                rep = atoms.info['key_value_pairs'][self.replica_key]
                traj = Trajectory(f'{self.calculation_dir}/tmp_{self.name}/emc_{rep}.traj', 'a')
                traj.write(atoms) 

            gi += 1
