from .gc_utils import compute_phi
from ase.geometry.analysis import Analysis
from ase.neighborlist import neighbor_list, natural_cutoffs
from ase.parallel import parprint
from ase.io import read, write
from ase.db import connect
from ase import units
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
import pickle
import os


class GEEMCAnalysis(object):

    """Analysis class for bulk GE-EMC simulations."""

    def __init__(self, name,
            calculation_dir,
            pop_size,
            target_metal_species,
            target_nonmetal_species=None,
            trim_generation=100,
            temperature_key='temperature',
            replica_key='replica',
            acceptance_key='accepted'):

        self.name = name
        self.calculation_dir = calculation_dir
        self.pop_size = pop_size
        self.target_metal_species = target_metal_species
        self.target_nonmetal_species = target_nonmetal_species
        self.trim_generation = trim_generation
        self.temperature_key = temperature_key
        self.replica_key = replica_key
        self.acceptance_key = acceptance_key

    def prepare_data(self, starting_step=None, ending_step=None, pressure=None,
                     use_exist=True, raw_score_cutoff=-10000.):

        """Prepare GE-EMC potential energy and Widom samples. Set the pressure 
        for NpT GE-EMC simulations"""

        collect_data = True
        if use_exist:
            if os.path.isfile(f'{self.calculation_dir}/{self.name}_mbar_widom_data.pkl') and os.path.isfile(
            f'{self.calculation_dir}/{self.name}_mbar_data.pkl'):                
                with open(f'{self.calculation_dir}/{self.name}_mbar_widom_data.pkl', 'rb') as handle:
                    self.widom_data = pickle.load(handle)
                with open(f'{self.calculation_dir}/{self.name}_mbar_data.pkl', 'rb') as handle:
                    self.data = pickle.load(handle)
                collect_data = False
            else:
                parprint(f'Cannot find MBAR data file, collect data instead.')
        if collect_data:
            # initialize containers
            self.widom_data = {}
            self.data = {}
            if starting_step is None:
                starting_step = 1
            rep_latest_step_map = {rep: 1 for rep in range(self.pop_size)}
            n = 0
            while os.path.isfile(f'{self.calculation_dir}/{self.name}_bak{n}.db'):
                n += 1
            for i in range(n):
                # skip entire backup if its block is before starting_step
                if ((i+1) * self.trim_generation) <= starting_step:
                    continue
                db = connect(f'{self.calculation_dir}/{self.name}_bak{i}.db')
                for j, row in enumerate(db.select('relaxed=1')):
                    if row.emc_step >= starting_step:
                        if row.raw_score <= raw_score_cutoff:
                            continue
                        if (ending_step is not None) and (row.emc_step > ending_step):
                            continue
                        # obtain temperature from the row and ensure containers exist for it
                        T = row.key_value_pairs[self.temperature_key]
                        if T not in self.widom_data:
                            self.widom_data[T] = {}
                        if T not in self.data:
                            self.data[T] = {}
                        dU = -row.delta
                        if abs(dU) > 100.:
                            continue
                        if abs(dU) < 0.01:
                            continue
                        atoms = row.toatoms()
                        U = row.potential_energy
                        is_widom = ((row.data.operation == 'interswap') and ('mutated_elements' in row.data))
                        if (row.key_value_pairs[self.acceptance_key] == 1) or is_widom:
                            atoms.info = {'key_value_pairs': {
                                              self.temperature_key: T, 
                                              self.replica_key: row.key_value_pairs[self.replica_key],
                                          'potential_energy': U}, 
                                          'data': {}}
                            rep_latest_step_map[row.key_value_pairs[self.replica_key]] = row.emc_step
                            if pressure is not None:
                                V = row.volume
                                U += pressure * V
                                atoms.info['key_value_pairs']['enthalpy'] = U
                                atoms.info['key_value_pairs']['NlnV'] = len(atoms) * np.log(V)
                            # ensure replica list exists and is long enough, then set energy
                            rep = row.key_value_pairs[self.replica_key]
                            rep_list = self.data[T].setdefault(rep, [])
                            needed = row.emc_step - len(rep_list)
                            if needed > 0:
                                rep_list.extend([None] * needed) 
                            rep_list[row.emc_step-1] = atoms

                        if not is_widom:
                            continue
                        m1, m2 = row.data.mutated_elements
                        N1 = len([a for a in atoms if a.symbol == m1]) + 1
                        N2plus1 = len([a for a in atoms if a.symbol == m2])
                        W_prefactor = N1 / N2plus1
                        if pressure is not None:
                            W_prefactor *= row.volume_ratio**len(atoms)
                        latest_acc_step = rep_latest_step_map[row.key_value_pairs[self.replica_key]]
                        # ensure dmu key list exists and is long enough, then append the tuple
                        key = f'{m1}2{m2}'
                        key_list = self.widom_data[T].setdefault(key, [])
                        needed = row.emc_step - len(key_list)
                        if needed > 0:
                            key_list.extend([[] for _ in range(needed)])  # distinct empty lists
                        key_list[row.emc_step-1].append(
                                (U, dU, W_prefactor, row.key_value_pairs[self.replica_key], latest_acc_step)
                        )

            # Assign the previous energy if energy is None at a step (forward-fill per replica)
            for T, dd in self.data.items():
                for rep in list(dd.keys()):
                    atoms_prev = None
                    newlist = []
                    for atoms in dd[rep]:
                        if atoms is not None:
                            atoms_prev = atoms
                            newlist.append(atoms)
                        else:
                            newlist.append(atoms_prev)
                    dd[rep] = newlist

            # save pickles
            with open(f'{self.calculation_dir}/{self.name}_mbar_widom_data.pkl', 'wb') as handle:
                pickle.dump(self.widom_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'{self.calculation_dir}/{self.name}_mbar_data.pkl', 'wb') as handle:
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_trajectory(self, replica):

        """Get the EMC trajectory of a given replica number."""

        if not hasattr(self, 'data'):
            raise RuntimeError('No trajectory data found, run prepare_data() first.')
        traj = []
        for T, dd in self.data.items():
            if replica in dd.keys():
                for atoms in dd[replica]:
                    if atoms is not None:
                        traj.append(atoms)

        write(f'{self.calculation_dir}/{self.name}_geemc_{replica}.traj', traj)

    def get_dmu_estimates(self, starting_step=None, ending_step=None, sample_stride=1, 
                          target_temperatures=None, volume_correction=False, wls_correction=True):

        """Get the MBAR-estimated chemical potential differences between all 
           metal pairs with/without WLS correction. Turn on volume_correction
           for NpT GE-EMC simulations."""

        from pymbar import MBAR
        if not hasattr(self, 'data'):
            raise RuntimeError('No trajectory data found, run prepare_data() first.')
        sim_temperatures = sorted(list(self.data.keys()))
        if target_temperatures is None:
            target_temperatures = sim_temperatures
        else:
            target_temperatures += sim_temperatures
            target_temperatures = sorted(set(target_temperatures))
        rep_temp_map = {}
        for T in sim_temperatures:
            for rep in self.data[T].keys():
                rep_temp_map[rep] = T
        metal_pairs = list(combinations(self.target_metal_species, 2))
        if self.target_nonmetal_species is None:
            nonmetal_pairs = []
        else:
            nonmetal_pairs = list(combinations(self.target_nonmetal_species, 2))

        K = len(target_temperatures)  # No. target of temperatures  
        # Build MBAR inputs  
        temp_step_idx_map = {T: {} for T in sim_temperatures}
        samples_per_T = np.zeros(K, dtype=int)  

        # Fill energy matrix and widom data
        if starting_step is None:
            starting_step = 1
        if ending_step is None:
            max_steps = 0
            for T, rep_dict in self.data.items():
                for rep, lst in rep_dict.items():
                    if lst:
                        max_steps = max(max_steps, len(lst))
            ending_step = max_step + 1
        parprint(f'Ending step: {ending_step}')

        betas = [1/(units.kB * T) for T in target_temperatures]
        samples = []
        volume_samples = []
        idx = 0
        for T in sim_temperatures:
            samples_T = []
            volume_samples_T = []
            for n_minus_1 in range(ending_step):
                if n_minus_1 + 1 >= starting_step:
                    try:
                        if volume_correction:
                            H = sum(lst[n_minus_1].info['key_value_pairs']['enthalpy'] for lst in self.data[T].values())
                            NlnV = sum(lst[n_minus_1].info['key_value_pairs']['NlnV'] for lst in self.data[T].values())
                        else:
                            U = sum(lst[n_minus_1].info['key_value_pairs']['potential_energy'] for lst in self.data[T].values())
                    except:
                        break
                    temp_step_idx_map[T][n_minus_1+1] = idx
                    if volume_correction:
                        samples_T.append(H)
                        volume_samples_T.append(NlnV)
                    else:
                        samples_T.append(U)
                    idx += 1
            samples_per_T[target_temperatures.index(T)] = len(samples_T)
            samples += samples_T
            if volume_correction:
                volume_samples += volume_samples_T
        samples = np.array(samples)
        U_kn = []  # Reduced energy matrix (K x N_total)
        NlnV_kn = []
        for k, T in enumerate(target_temperatures):
            U_kn.append(betas[k] * samples)
            if volume_correction:
                NlnV_kn.append(volume_samples)
        U_kn = np.array(U_kn)  # Shape (K, N_total)
        if volume_correction:
            U_kn -= np.array(NlnV_kn)
        mbar = MBAR(U_kn, samples_per_T, relative_tolerance=1e-7, 
                    solver_protocol='robust', maximum_iterations=30000)  
        w_nk = mbar.weights()

        # Compute Δμ_{m→n} for each pair
        dmu_estimates = {T: {} for T in target_temperatures}
        for T in target_temperatures:
            for m1, m2 in metal_pairs + nonmetal_pairs:
                dmu_estimates[T][f'{m1}2{m2}'] = {}
                dmu_estimates[T][f'{m2}2{m1}'] = {}
        for m1, m2 in metal_pairs + nonmetal_pairs:
            # Precompute W_{m→n} for ALL configs, replicas, and target betas
            lot_m12m2 = [t for T in sim_temperatures for lot in self.widom_data[T][f'{m1}2{m2}'] for t in lot]
            dd_temp_m12m2 = {T: defaultdict(list) for T in target_temperatures}
            for U_new, dU, W_prefactor, rep, latest_acc_step in lot_m12m2:
                if latest_acc_step > starting_step:
                    T_origin = rep_temp_map[rep]
                    for T in target_temperatures:
                        W = W_prefactor * np.exp(-dU / (units.kB*T))
                        dd_temp_m12m2[T][(T_origin, latest_acc_step)].append(W)
            lot_m22m1 = [t for T in sim_temperatures for lot in self.widom_data[T][f'{m2}2{m1}'] for t in lot]
            dd_temp_m22m1 = {T: defaultdict(list) for T in target_temperatures}
            for U_new, dU, W_prefactor, rep, latest_acc_step in lot_m22m1:
                if latest_acc_step > starting_step:
                    T_origin = rep_temp_map[rep]
                    for T in target_temperatures:
                        W = W_prefactor * np.exp(-dU / (units.kB*T))
                        dd_temp_m22m1[T][(T_origin, latest_acc_step)].append(W)

            # SUBSAMPLE BY SIMULATION-STEP BIN WIDTH = sample_stride
            # Keep the first sample in each integer bin of width `sample_stride`
            # (binning applied separately for each T_origin).
            if (sample_stride is not None) and (sample_stride > 1):
                def subsample_dd_by_step(dd_temp):
                    dd_sub = {T: defaultdict(list) for T in target_temperatures}
                    for T in target_temperatures:
                        # group keys by T_origin
                        keys = list(dd_temp[T].keys())  # each key = (T_origin, step)
                        by_origin = defaultdict(list)
                        for (T_origin, step) in keys:
                            by_origin[T_origin].append(step)
                        # for each origin, bin steps and pick representative step in each bin
                        for T_origin, steps in by_origin.items():
                            steps_arr = np.array(sorted(steps))
                            if steps_arr.size == 0:
                                continue
                            # compute bins relative to starting_step so bins align with emc steps
                            bins = ((steps_arr - starting_step) // int(sample_stride)).astype(int)
                            # pick first sample in each unique bin
                            #_, first_idx = np.unique(bins, return_index=True)
                            #keep_steps = steps_arr[first_idx]
                            # pick last sample in each unique bin
                            unique_bins, last_pos_rev = np.unique(bins[::-1], return_index=True)
                            last_pos = bins.size - 1 - last_pos_rev
                            keep_steps = steps_arr[last_pos]
                            for step in keep_steps:
                                dd_sub[T][(T_origin, int(step))] = dd_temp[T][(T_origin, int(step))]
                    return dd_sub

                dd_temp_m12m2 = subsample_dd_by_step(dd_temp_m12m2)
                dd_temp_m22m1 = subsample_dd_by_step(dd_temp_m22m1)

            # Compute valid ids and MBAR expectations
            valid_m12m2_ids, valid_m22m1_ids = [], []
            valid_W_m12m2, valid_W_m22m1 = [[] for _ in range(K)], [[] for _ in range(K)]
            valid_w_m12m2_kn, valid_w_m22m1_kn = [[] for _ in range(K)], [[] for _ in range(K)]
            Wbar_m12m2_lol = [[None for _ in range(U_kn.shape[1])] for _ in range(K)]
            Wbar_m22m1_lol = [[None for _ in range(U_kn.shape[1])] for _ in range(K)]
            valid_m12m2_samples_per_T = np.zeros(K, dtype=int)
            valid_m22m1_samples_per_T = np.zeros(K, dtype=int)

            for k, T in enumerate(target_temperatures):
                for tup, W_lst in dd_temp_m12m2[T].items():
                    T_origin, n = tup
                    try:
                        idx = temp_step_idx_map[T_origin][n]
                    except:
                        continue
                    weights = np.full(len(W_lst), w_nk[idx,k] / len(W_lst)).tolist()
                    valid_W_m12m2[k] += W_lst
                    valid_w_m12m2_kn[k] += weights
                    if k == 0:
                        valid_m12m2_ids.append(idx)
                    Wbar_m12m2_lol[k][idx] = np.mean(W_lst) # Average W per sample
                    if T == T_origin:
                        # Count the samples from origin T
                        valid_m12m2_samples_per_T[target_temperatures.index(T_origin)] += 1 
                for tup, W_lst in dd_temp_m22m1[T].items():
                    T_origin, n = tup
                    try:
                        idx = temp_step_idx_map[T_origin][n]
                    except:
                        continue
                    weights = np.full(len(W_lst), w_nk[idx,k] / len(W_lst)).tolist()
                    valid_W_m22m1[k] += W_lst
                    valid_w_m22m1_kn[k] += weights
                    if k == 0:
                        valid_m22m1_ids.append(idx)
                    Wbar_m22m1_lol[k][idx] = np.mean(W_lst) # Average W per sample
                    if T == T_origin:
                        # Count the samples from origin T
                        valid_m22m1_samples_per_T[target_temperatures.index(T_origin)] += 1 

            valid_m12m2_ids = [i for i, Wbar in enumerate(Wbar_m12m2_lol[0]) if Wbar is not None]
            valid_m22m1_ids = [i for i, Wbar in enumerate(Wbar_m22m1_lol[0]) if Wbar is not None]
            Wbar_m12m2_arr = np.array(Wbar_m12m2_lol, dtype=float)[:,valid_m12m2_ids]
            Wbar_m22m1_arr = np.array(Wbar_m22m1_lol, dtype=float)[:,valid_m22m1_ids]
            # Reweighting at temperature T
            mbar_m12m2 = MBAR(U_kn[:,valid_m12m2_ids], valid_m12m2_samples_per_T, 
                              relative_tolerance=1e-7, solver_protocol='robust', maximum_iterations=30000)
            # Compute expectation with MBAR
            res_m12m2 = mbar_m12m2.compute_expectations(Wbar_m12m2_arr, compute_uncertainty=True, state_dependent=True)
            E_W_m12m2, sigma_W_m12m2 = res_m12m2['mu'], res_m12m2['sigma']
            #W_mbar_m12m2 = mbar_m12m2.weights()
            mbar_m22m1 = MBAR(U_kn[:,valid_m22m1_ids], valid_m22m1_samples_per_T, 
                              relative_tolerance=1e-7, solver_protocol='robust', maximum_iterations=30000)  
            # Compute expectation with MBAR                                                                   
            res_m22m1 = mbar_m22m1.compute_expectations(Wbar_m22m1_arr, compute_uncertainty=True, state_dependent=True)
            E_W_m22m1, sigma_W_m22m1 = res_m22m1['mu'], res_m22m1['sigma']

            for k, T in enumerate(target_temperatures):
                beta_target = 1/(units.kB * T)
                E_W_m12m2_T, Var_W_m12m2_T = E_W_m12m2[k], sigma_W_m12m2[k]**2
                E_W_m22m1_T, Var_W_m22m1_T = E_W_m22m1[k], sigma_W_m22m1[k]**2
                # Delta method
                Var_dmu_m12m2_T = Var_W_m12m2_T / (beta_target * E_W_m12m2_T)**2
                Var_dmu_m22m1_T = Var_W_m22m1_T / (beta_target * E_W_m22m1_T)**2
                dmu_m12m2_T = -(1/beta_target) * np.log(E_W_m12m2_T)
                dmu_m22m1_T = -(1/beta_target) * np.log(E_W_m22m1_T)
                w_m12m2_T = 1 / Var_dmu_m12m2_T
                w_m22m1_T = 1 / Var_dmu_m22m1_T
                dmu_estimates[T][f'{m1}2{m2}']['weight'] = w_m12m2_T
                dmu_estimates[T][f'{m1}2{m2}']['estimate'] = dmu_m12m2_T
                dmu_estimates[T][f'{m2}2{m1}']['weight'] = w_m22m1_T
                dmu_estimates[T][f'{m2}2{m1}']['estimate'] = dmu_m22m1_T

        if wls_correction:
            M = len(self.target_metal_species)
            temperatures = sorted(dmu_estimates.keys())
            wls_dmu_estimates = {T: {} for T in temperatures}
            for T, dd in dmu_estimates.items():
                # Initialize Laplacian and RHS
                L = np.zeros((M, M))
                b = np.zeros(M)
                # iterate over the raw directed entries present in dd
                # assume keys are like 'Co2Ni' representing Co -> Ni
                for key, info in dd.items():
                    # parse species
                    m1, m2 = key.split('2')
                    if (m1 not in self.target_metal_species) or (m2 not in self.target_metal_species):
                        continue
                    i = self.target_metal_species.index(m1)
                    j = self.target_metal_species.index(m2)
                    # get the directed estimate d_k
                    w = info.get('weight')
                    dmu_raw = info.get('estimate')
                    if dmu_raw is None:
                        continue
                    # Update Laplacian L and RHS b
                    L[i, i] += w
                    L[j, j] += w
                    L[i, j] -= w
                    L[j, i] -= w
                    # Update RHS: add contributions for directed eq mu_j - mu_i = dmu_raw
                    b[i] -= w * dmu_raw
                    b[j] += w * dmu_raw
            
                # Reference constraint μ_Co = 0  (keep same reference index)
                ref_idx = 0
                keep_mask = np.arange(M) != ref_idx
                L_red = L[keep_mask, :][:, keep_mask]
                b_red = b[keep_mask] - L[keep_mask, ref_idx] * 0.0
                # Solve reduced system robustly
                try:
                    mu_red = np.linalg.solve(L_red, b_red)
                except np.linalg.LinAlgError:
                    mu_red = np.linalg.lstsq(L_red, b_red, rcond=None)[0]
                # combine into full mu vector
                mu = np.zeros(M)
                mu[ref_idx] = 0.0
                mu[keep_mask] = mu_red
            
                # produce corrected differences for all unordered pairs
                for m1, m2 in combinations(self.target_metal_species, 2):
                    i = self.target_metal_species.index(m1)
                    j = self.target_metal_species.index(m2)
                    wls_dmu_estimates[T][f'{m1}2{m2}'] = mu[j] - mu[i]
            dmu_estimates = wls_dmu_estimates
                    
        with open(f'{self.calculation_dir}/{self.name}_mbar_dmu_estimates.pkl', 'wb') as handle:
            pickle.dump(dmu_estimates, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return dmu_estimates

    def get_sro(self, starting_step=None, ending_step=None, sample_stride=1,
                temperature=None, cutoff=None, metal_only=True, use_energy_mask=False):
        """
        Compute composite Warren-Cowley pairwise multicomponent SRO at each step:
        alpha_{mn}(R) = (1/R) sum_r [ (P_mn(r) - c_n(R)) / (delta_mn - c_n(R)) ].

        starting_step, ending_step, sample_stride: subsampling EMC steps 
        temperature: if provided, analyze only that T, otherwise all Ts.
        cutoff: neighbor cutoff for first shell, use natural_cutoffs by default.
        use_energy_mask: if True, skip running neighbor_list for replicas whose
            potential energy equals that at the previous step for acceleration.
        """

        if not hasattr(self, 'data'):
            raise RuntimeError('No trajectory data found, run prepare_data() first.')
        # build target species lists (exclude vacancies 'X')
        metals = [m for m in (self.target_metal_species or []) if m != 'X']
        nonmetals = [n for n in (self.target_nonmetal_species or []) if n != 'X']
        if metal_only:
            elem_list = metals
        else:
            elem_list = metals + nonmetals
        # unordered hetero pairs + homo pairs
        pairs = list(combinations(elem_list, 2)) + [(e, e) for e in elem_list]
        sim_temps = sorted(self.data.keys())
        if temperature is None:
            temps = sim_temps
        else:
            if temperature not in sim_temps:
                raise KeyError(f'Temperature {temperature} not found in data.')
            temps = [temperature]

        # per-temperature SRO data collection
        sro_data = {}
        for T in temps:
            rep_dict = self.data[T]
            # step bounds aligned by replica lists length
            max_steps = max((len(lst) for lst in rep_dict.values()), default=0)
            s0 = 1 if starting_step is None else int(starting_step)
            s1 = max_steps if ending_step is None else int(min(ending_step, max_steps))
            if s1 < s0:
                raise ValueError('Invalid step range or no steps in requested range.')
            steps = list(range(s0, s1 + 1, int(sample_stride)))
            # prepare output dict: for each pair, a list of composite alphas
            pair_step_alpha = {pair: [] for pair in pairs}
            last_neigh_info = {}
            for step in steps:
                reps_atoms = []
                for rep, alist in rep_dict.items():
                    if len(alist) >= step and alist[step - 1] is not None:
                        reps_atoms.append((rep, alist[step - 1]))
                R = len(reps_atoms)
                if R == 0:
                    # no data at this step
                    for pair in pairs:
                        pair_step_alpha[pair].append(np.nan)
                    continue

                # 1) build or reuse per-replica counts and neighbor histograms
                rep_neigh_info = {}
                for rep, atoms in reps_atoms:
                    energy = None
                    try:
                        energy = atoms.info['key_value_pairs'].get('potential_energy', None)
                    except Exception:
                        energy = None
                    reuse = False
                    if use_energy_mask and (rep in last_neigh_info):
                        prev_energy = last_neigh_info[rep].get('energy', None)
                        if (prev_energy is not None) and (energy is not None) and (energy == prev_energy):
                            # reuse cached neighbor info and counts (skip neighbor_list)
                            rep_neigh_info[rep] = last_neigh_info[rep]
                            reuse = True
                    if not reuse:
                        # compute counts and per-reference neighbor histograms using neighbor_list
                        syms = np.array(atoms.get_chemical_symbols())
                        # counts for composite concentration: only targeted species and ignore 'X'
                        filtered_syms = syms[(syms != 'X') & np.isin(syms, elem_list)]
                        counts = Counter(filtered_syms.tolist())
                        # neighbor_list gives directed pairs i->j within cutoff
                        if cutoff is None:
                            cutoff_local = natural_cutoffs(atoms)
                        else:
                            cutoff_local = cutoff
                        i_idx, j_idx = neighbor_list('ij', atoms, cutoff_local)
                        if i_idx.size == 0:
                            # empty neighbor structure
                            per_ref = {}
                        else:
                            si = syms[i_idx]
                            sj = syms[j_idx]
                            per_ref = {}
                            # for each reference species m we collect neighbors around all m atoms
                            for m in elem_list:
                                mask_i = (si == m)
                                if not np.any(mask_i):
                                    # no reference atoms of this species in this replica
                                    continue
                                neigh_syms = sj[mask_i]
                                # filter neighbors to targeted species only
                                if neigh_syms.size == 0:
                                    continue
                                keep = np.isin(neigh_syms, elem_list)
                                neigh_syms = neigh_syms[keep]
                                total_neigh = neigh_syms.size
                                if total_neigh == 0:
                                    continue
                                uniq, cnts = np.unique(neigh_syms, return_counts=True)
                                neigh_map = dict(zip(uniq.tolist(), cnts.tolist()))
                                per_ref[m] = (int(total_neigh), neigh_map)

                        cache_info = {'energy': energy, 'counts': counts, 'per_ref': per_ref}
                        rep_neigh_info[rep] = cache_info
                        last_neigh_info[rep] = cache_info

                # 2) compute composite concentrations c_n(R) using rep_neigh_info counts
                total_counts = Counter()
                for rep, info in rep_neigh_info.items():
                    total_counts.update(info.get('counts', Counter()))
                total_metal_count = sum(total_counts[e] for e in metals) if metals else 0
                total_nonmetal_count = sum(total_counts[e] for e in nonmetals) if nonmetals else 0
                cR = {}
                for e in elem_list:
                    if e in metals:
                        if total_metal_count > 0:
                            cR[e] = (total_counts.get(e, 0) / total_metal_count)
                        else:
                            cR[e] = np.nan
                    else:
                        if total_nonmetal_count > 0:
                            cR[e] = (total_counts.get(e, 0) / total_nonmetal_count)
                        else:
                            cR[e] = np.nan

                # 3) for every pair compute average SRO over composite replicas
                for pair in pairs:
                    m, n = pair
                    alpha_sum = 0.
                    valid_count = 0
                    for rep, info in rep_neigh_info.items():
                        per_ref = info.get('per_ref', {})
                        if m not in per_ref:
                            # no reference atoms of type m in this replica -> undefined
                            continue
                        total_neigh, neigh_map = per_ref[m]
                        if total_neigh == 0:
                            continue
                        count_n = neigh_map.get(n, 0)
                        P_mn_r = float(count_n) / float(total_neigh)
                        c_n = cR.get(n, np.nan)
                        delta = 1. if m == n else 0.
                        denom = delta - c_n
                        if denom == 0 or np.isnan(denom):
                            continue
                        alpha_r = (P_mn_r - c_n) / denom
                        alpha_sum += alpha_r
                        valid_count += 1
                    if valid_count == 0:
                        pair_step_alpha[pair].append(np.nan)
                    else:
                        pair_step_alpha[pair].append(float(alpha_sum / float(valid_count)))
            sro_data[T] = pair_step_alpha

        # save pickle
        fname = f'{self.calculation_dir}/{self.name}_sro_data.pkl' if temperature is None \
                else f'{self.calculation_dir}/{self.name}_sro_data_{temperature}K.pkl'
        with open(fname, 'wb') as handle:
            pickle.dump(sro_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return sro_data


class GCEMCAnalysis(object):

    def __init__(self, name,
            calculation_dir,
            pop_size,
            target_metal_species,
            target_metal_composition,
            adsorbate_species,
            trim_generation=100,
            temperature_key='temperature',
            replica_key='replica',
            acceptance_key='accepted'):

        self.name = name
        self.calculation_dir = calculation_dir
        self.pop_size = pop_size
        self.target_metal_species = target_metal_species
        self.target_metal_composition = target_metal_composition
        self.adsorbate_species = adsorbate_species
        self.trim_generation = trim_generation
        self.temperature_key = temperature_key
        self.replica_key = replica_key
        self.acceptance_key = acceptance_key

    def prepare_data(self, starting_step=None, ending_step=None, 
                     use_exist=True, raw_score_cutoff=-10000.):

        """Prepare GC-EMC potential energy samples."""

        collect_data = True
        if use_exist:
            if os.path.isfile(f'{self.calculation_dir}/{self.name}_mbar_data.pkl'):
                with open(f'{self.calculation_dir}/{self.name}_mbar_data.pkl', 'rb') as handle:
                    self.data = pickle.load(handle)
                collect_data = False
            else:
                parprint(f'Cannot find MBAR data file, collect data instead.')
        if collect_data:
            # initialize containers
            self.data = {} 
            if starting_step is None:
                starting_step = 1
            rep_latest_step_map = {rep: 1 for rep in range(self.pop_size)}
            n = 0
            while os.path.isfile(f'{self.calculation_dir}/{self.name}_bak{n}.db'):
                n += 1
            for i in range(n):
                # skip entire backup if its block is before starting_step
                if ((i+1) * self.trim_generation) <= starting_step:
                    continue
                db = connect(f'{self.calculation_dir}/{self.name}_bak{i}.db')
                for j, row in enumerate(db.select('relaxed=1')):
                    if row.emc_step >= starting_step:
                        if row.raw_score <= raw_score_cutoff:
                            continue
                        if (ending_step is not None) and (row.emc_step > ending_step):
                            continue
                        # obtain temperature from the row and ensure containers exist for it
                        T = row.key_value_pairs[self.temperature_key]
                        if T not in self.data:
                            self.data[T] = {}

                        if row.key_value_pairs[self.acceptance_key] == 1:
                            atoms = row.toatoms()
                            atoms.info = {'key_value_pairs': {
                                              self.temperature_key: T, 
                                              self.replica_key: row.key_value_pairs[self.replica_key],
                                              'potential_energy': row.potential_energy}, 
                                          'data': {}}
                            atoms.info['data']['mu'] = row.data.get('mu', None)
                            atoms.info['data']['mu_window'] = row.data.get('mu_window', None)
                            atoms.info['data']['fragments'] = row.data.get('fragments', None)
                            # ensure replica list exists and is long enough, then set energy
                            rep = row.key_value_pairs[self.replica_key]
                            rep_list = self.data[T].setdefault(rep, [])
                            needed = row.emc_step - len(rep_list)
                            if needed > 0:
                                rep_list.extend([None] * needed)  # None is immutable, multiplication is safe
                            rep_list[row.emc_step-1] = atoms

            # Assign the previous energy if energy is None at a step (forward-fill per replica)
            for T, dd in self.data.items():
                for rep in list(dd.keys()):
                    atoms_prev = None
                    newlist = []
                    for atoms in dd[rep]:
                        if atoms is not None:
                            atoms_prev = atoms
                            newlist.append(atoms)
                        else:
                            newlist.append(atoms_prev)
                    dd[rep] = newlist

            # save pickle
            with open(f'{self.calculation_dir}/{self.name}_mbar_data.pkl', 'wb') as handle:
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_trajectory(self, replica):

        """Get the EMC trajectory of a given replica number."""

        if not hasattr(self, 'data'):
            raise RuntimeError('No trajectory data found, run prepare_data() first.')
        traj = []
        for T, dd in self.data.items():
            if replica in dd.keys():
                for atoms in dd[replica]:
                    if atoms is not None:
                        traj.append(atoms)

        write(f'{self.calculation_dir}/{self.name}_gcemc_{replica}.traj', traj)

    def get_structural_expectations(self, z_surf, reaction_conditions, element_column_map, 
                                    element_eref_map, element_y2x_map, temperature_dmu_map=None, 
                                    starting_step=None, ending_step=None, sample_stride=1, 
                                    sample_condition_indices=None, vacancy_indices=None, tol=1e-3):

        """Get the MBAR-estimated expectations of surface structural properties 
        depending on a user-defined surface region (> z_surf) and reaction conditions. 
        The reaction conditions is supposed to emcompass all the reaction condition
        grid considered in the EMC simulation, including last column as temperature.
        You can also add more rows to reaction_conditions to consider states that are
        not included in the simulation."""

        if not hasattr(self, 'data'):
            raise RuntimeError('No trajectory data found, run prepare_data() first.')
        from pymbar import MBAR, timeseries
        sim_temperatures = sorted(list(self.data.keys()))
        nonmetal_species = list(element_column_map.keys()) if element_column_map else []
        all_species = list(self.target_metal_species) + nonmetal_species
        metal_pairs = list(combinations(self.target_metal_species, 2))
        all_atom_pairs = list(combinations(all_species, 2)) + [(m,m) for m in all_species]
        if vacancy_indices is None:
            vacancy_indices = []
        if reaction_conditions is None:
            grid = np.array(sim_temperatures)
        else:
            grid = reaction_conditions.copy()
        K = grid.shape[0]
        parprint(f'Number of states: {K}')
        
        # Build MBAR inputs
        samples_per_state = np.zeros(K, dtype=int)
        # Fill energy matrix
        phi_kn, images = [], []
        sample_state = []  # length = total number of appended samples, gives state index for each column in phi_kn
        sample_steps = []   # record the original simulation step (n_minus_1) for each appended sample
        for k, T in enumerate(sim_temperatures):
            for rep, lol in self.data[T].items():
                for n_minus_1, atoms in enumerate(lol):
                    if (starting_step is None) or (n_minus_1 + 1 >= starting_step):
                        if (ending_step is not None) and (n_minus_1 + 1 > ending_step):
                            continue
                        if atoms is None:
                            continue
                        if not os.path.isfile(f'{self.calculation_dir}/{self.name}_mbar.traj'):
                            surf_num_dict = Counter(atoms[[a.index for a in atoms if
                                    (a.position[2] > z_surf) and (a.symbol in self.target_metal_species)]].symbols)
                            n_surf_metals = sum(surf_num_dict.values())
                            for m in self.target_metal_species:
                                atoms.info['data'][f'x_surf_{m}'] = surf_num_dict[m] / n_surf_metals
                            atoms.info['data']['x_surf_metals'] = np.array(
                                    [surf_num_dict[m] / n_surf_metals for m in self.target_metal_species])
                            fragments = atoms.info['data']['fragments']
                            lattice_nonmetal_symbols = []
                            for nm in element_column_map.keys():
                                lattice_nonmetal_symbols += [nm for i in vacancy_indices if atoms[i].symbol == nm]
                            adsorbates = list((Counter(fragments) - Counter(lattice_nonmetal_symbols)).elements())
                            ads_num_dict = Counter(adsorbates)
                            atoms.info['data']['ads_coverages'] = np.array([ads_num_dict[ads] / n_surf_metals for ads in self.adsorbate_species])
                            atoms.info['data']['ads_coverage'] = len(adsorbates) / n_surf_metals
                            for ads in self.adsorbate_species:
                                if ads in ads_num_dict:
                                    atoms.info['data'][f'ads_coverage_{ads}'] = ads_num_dict[ads] / n_surf_metals
                                else:
                                    atoms.info['data'][f'ads_coverage_{ads}'] = 0.
                            if vacancy_indices:
                                x_vac = len([a for a in atoms if (a.symbol == 'X') and (a.position[2] < z_surf)]) / len(vacancy_indices)
                                atoms.info['data'][f'x_lattice_vac'] = x_vac
                            surf_atoms = atoms[[a.index for a in atoms if (a.symbol != 'X') and (a.position[2] > z_surf)]]
                            surf_atoms.pbc = [1,1,0]
                            surf_num_dict = Counter(surf_atoms.symbols)
                            cutoffs = natural_cutoffs(surf_atoms, mult=1.1)
                            ana = Analysis(surf_atoms, cutoffs=cutoffs)
                            surf_atom_pairs = []
                            for i, nblst in enumerate(ana.unique_bonds[0]):
                                for j in nblst:
                                    surf_atom_pairs.append(set([surf_atoms[i].symbol, surf_atoms[j].symbol]))
                            for pair in all_atom_pairs:
                                m1, m2 = pair
                                atoms.info['data'][f'{m1}-{m2}'] = len([s for s in surf_atom_pairs if s == set(pair)])
                            images.append(atoms)
 
                        epot = atoms.info['key_value_pairs']['potential_energy']
                        num_metal = len([a for a in atoms if a.symbol in self.target_metal_species])
                        num_dict = dict((m, len([a for a in atoms if a.symbol==m])) for m in all_species)
 
                        sample_cond = np.append(atoms.info['data']['mu'], T)
                        idx_match = np.where(np.all(np.abs(grid-sample_cond) < tol, axis=1))[0]
                        if idx_match.size == 0:
                            parprint('No matching state found for sample:', sample_cond)
                            continue
                        state_idx = int(idx_match[0])
                        samples_per_state[state_idx] += 1
                        sample_state.append(state_idx)
                        sample_steps.append(n_minus_1)
                        phi = compute_phi(
                                num_dict, self.target_metal_species, self.target_metal_composition, 
                                temperature_dmu_map, grid, element_column_map, element_eref_map, 
                                element_y2x_map, num_metal, epot, dimensionless=True)
                        phi_kn.append(phi.reshape(-1,1))

        if os.path.isfile(f'{self.calculation_dir}/{self.name}_mbar.traj'):
            images = read(f'{self.calculation_dir}/{self.name}_mbar.traj', index=':')
        else:
            write(f'{self.calculation_dir}/{self.name}_mbar.traj', images)
 
        phi_kn = np.hstack(phi_kn)  # Shape (K, N_total)
        # Build numpy array of which state produced each sample (global column index)
        sample_state = np.asarray(sample_state, dtype=int)  # shape (N_total,)
        sample_steps = np.asarray(sample_steps, dtype=int)  # shape (N_total,)
        N_total = sample_state.size
        kept_indices_list = []
        present_states, present_counts = np.unique(sample_state, return_counts=True)
        parprint(f"Found {present_states.size} occupied states (out of {K}). Subsampling each occupied state's series...")

        for s, n_k in zip(present_states, present_counts):
            idx_k = np.where(sample_state == s)[0]     # global indices of samples from state s (chronological)
            if n_k == 0:
                continue
            if sample_stride is not None:
                # Interpret stride as simulation-step bin width: use original simulation steps (sample_steps)
                stride_int = int(sample_stride)
                steps_k = sample_steps[idx_k]           # original n_minus_1 values for samples of this state
                bins = steps_k // stride_int
                # pick the first sample that appears in each bin (preserves chronology)
                #unique_bins, first_pos = np.unique(bins, return_index=True)
                #subs_global = idx_k[first_pos]           # convert to global indices
                # take last occurrence per bin instead of first
                unique_bins, last_pos_rev = np.unique(bins[::-1], return_index=True)
                last_pos = bins.size - 1 - last_pos_rev
                subs_global = idx_k[last_pos]
            else:
                # Let pymbar compute statistical inefficiency and return per-state local indices
                series = phi_kn[int(s), idx_k].ravel()
                try:
                    local_idx = np.array(timeseries.subsample_correlated_data(series))  # indices into series
                    subs_global = idx_k[local_idx]
                except Exception as e:
                    parprint(f'  warning: subsample_correlated_data failed for state {s}: {e}; keeping all samples for this state')
                    subs_global = idx_k.copy()
            if subs_global.size == 0:
                parprint(f'  state {s}: no subsampled indices -> skipping')
                continue
            kept_indices_list.append(subs_global)
            parprint(f' state {s}: n_k={n_k}, kept={subs_global.size}')
 
        if len(kept_indices_list) == 0:
            uniq, counts = np.unique(sample_state, return_counts=True)
            raise RuntimeError(f'No subsampled indices found for any occupied state. ' + 
                               f'Present states: {uniq}, counts: {counts}. phi_kn.shape={phi_kn.shape}')
        kept_indices = np.unique(np.concatenate(kept_indices_list)).astype(int)
        kept_indices.sort()
        np.save(f'{self.calculation_dir}/{self.name}_sampled_indices.npy', kept_indices)
        parprint(f'Original total samples: {N_total}, subsampled total: {kept_indices.size}')
 
        # make subsampled phi_kn and per-state counts for MBAR
        phi_kn_sub = phi_kn[:, kept_indices]
        samples_per_state_sub = np.array([np.sum(sample_state[kept_indices] == k) for k in range(K)], dtype=int)
        #parprint(f'New per-state sample counts (subsampled): {samples_per_state_sub}')
        images_sub = [images[i] for i in kept_indices]
 
        # Compute expectation with MBAR
        mbar = MBAR(phi_kn_sub, samples_per_state_sub, relative_tolerance=1e-7,
                    solver_protocol='robust', maximum_iterations=30000)
        w_nk = mbar.weights()
        w_nk /= w_nk.sum(axis=0)
 
        expectations = {}
        for cv in images_sub[0].info['data'].keys():
            if cv in ['fragments', 'x_surf_metals', 'ads_coverages', 'raw_scores', 'mu', 'mu_window']:
                continue
            A_n = np.array([a.info['data'][cv] for a in images_sub])
            if sample_condition_indices is None:
                E_k = w_nk.T @ A_n
            else:
                E_k = w_nk[:,sample_condition_indices].T @ A_n
            expectations[cv] = E_k
 
        with open(f'{self.calculation_dir}/{self.name}_mbar_expectations.pkl', 'wb') as handle:
            pickle.dump(expectations, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return expectations

    def get_structural_fes(self, z_surf, reaction_conditions, element_column_map,
                           element_eref_map, element_y2x_map, temperature_dmu_map=None,
                           starting_step=None, ending_step=None, sample_stride=1,
                           sample_condition_indices=None, vacancy_indices=None, 
                           variable_keys=None, bin_width=0.05, tol=1e-3):

        """Get the MBAR-estimated free energy surfaces of surface structural properties
        depending on a user-defined surface region (> z_surf), reaction conditions and
        a bin width. The reaction conditions is supposed to emcompass all the reaction 
        condition grid considered in the EMC simulation, including last column as 
        temperature. You can also add more rows to reaction_conditions to consider 
        states that are not included in the simulation."""

        if not hasattr(self, 'data'):
            raise RuntimeError('No trajectory data found, run prepare_data() first.')
        from pymbar import MBAR, FES, timeseries
        sim_temperatures = sorted(list(self.data.keys()))
        nonmetal_species = list(element_column_map.keys()) if element_column_map else []
        all_species = list(self.target_metal_species) + nonmetal_species
        metal_pairs = list(combinations(self.target_metal_species, 2))
        all_atom_pairs = list(combinations(all_species, 2)) + [(m,m) for m in all_species]
        if vacancy_indices is None:
            vacancy_indices = []
        if variable_keys is None:
            variable_keys = ['all']
        grid = reaction_conditions.copy()
        K = grid.shape[0]
        parprint(f'Number of states: {K}')

        # Build MBAR inputs
        samples_per_state = np.zeros(K, dtype=int)
        # Fill energy matrix
        phi_kn, images = [], []
        sample_state = []  # length = total number of appended samples, gives state index for each column in phi_kn
        sample_steps = []   # record the original simulation step (n_minus_1) for each appended sample
        for k, T in enumerate(sim_temperatures):
            for rep, lol in self.data[T].items():
                for n_minus_1, atoms in enumerate(lol):
                    if (starting_step is None) or (n_minus_1 + 1 >= starting_step):
                        if (ending_step is not None) and (n_minus_1 + 1 > ending_step):
                            continue
                        if atoms is None:
                            continue
                        if not os.path.isfile(f'{self.calculation_dir}/{self.name}_mbar.traj'):
                            surf_num_dict = Counter(atoms[[a.index for a in atoms if
                                    (a.position[2] > z_surf) and (a.symbol in self.target_metal_species)]].symbols)
                            n_surf_metals = sum(surf_num_dict.values())
                            for m in self.target_metal_species:
                                atoms.info['data'][f'x_surf_{m}'] = surf_num_dict[m] / n_surf_metals
                            atoms.info['data']['x_surf_metals'] = np.array(
                                    [surf_num_dict[m] / n_surf_metals for m in self.target_metal_species])
                            fragments = atoms.info['data']['fragments']
                            lattice_nonmetal_symbols = []
                            for nm in element_column_map.keys():
                                lattice_nonmetal_symbols += [nm for i in vacancy_indices if atoms[i].symbol == nm]
                            adsorbates = list((Counter(fragments) - Counter(lattice_nonmetal_symbols)).elements())
                            ads_num_dict = Counter(adsorbates)
                            atoms.info['data']['ads_coverages'] = np.array([ads_num_dict[ads] / n_surf_metals for ads in self.adsorbate_species])
                            atoms.info['data']['ads_coverage'] = len(adsorbates) / n_surf_metals
                            for ads in self.adsorbate_species:
                                if ads in ads_num_dict:
                                    atoms.info['data'][f'ads_coverage_{ads}'] = ads_num_dict[ads] / n_surf_metals
                                else:
                                    atoms.info['data'][f'ads_coverage_{ads}'] = 0.
                            if vacancy_indices:
                                x_vac = len([a for a in atoms if (a.symbol == 'X') and (a.position[2] < z_surf)]) / len(vacancy_indices)
                                atoms.info['data'][f'x_lattice_vac'] = x_vac
                            surf_atoms = atoms[[a.index for a in atoms if (a.symbol != 'X') and (a.position[2] > z_surf)]]
                            surf_atoms.pbc = [1,1,0]
                            surf_num_dict = Counter(surf_atoms.symbols)
                            cutoffs = natural_cutoffs(surf_atoms, mult=1.1)
                            ana = Analysis(surf_atoms, cutoffs=cutoffs)
                            surf_atom_pairs = []
                            for i, nblst in enumerate(ana.unique_bonds[0]):
                                for j in nblst:
                                    surf_atom_pairs.append(set([surf_atoms[i].symbol, surf_atoms[j].symbol]))
                            for pair in all_atom_pairs:
                                m1, m2 = pair
                                atoms.info['data'][f'{m1}-{m2}'] = len([s for s in surf_atom_pairs if s == set(pair)])
                            images.append(atoms)

                        epot = atoms.info['key_value_pairs']['potential_energy']
                        num_metal = len([a for a in atoms if a.symbol in self.target_metal_species])
                        num_dict = dict((m, len([a for a in atoms if a.symbol==m])) for m in all_species)

                        sample_cond = np.append(atoms.info['data']['mu'], T)
                        idx_match = np.where(np.all(np.abs(grid-sample_cond) < tol, axis=1))[0]
                        if idx_match.size == 0:
                            parprint('No matching state found for sample:', sample_cond)
                            continue
                        state_idx = int(idx_match[0])
                        samples_per_state[state_idx] += 1
                        sample_state.append(state_idx)
                        sample_steps.append(n_minus_1)
                        phi = compute_phi(
                                num_dict, self.target_metal_species, self.target_metal_composition,
                                temperature_dmu_map, grid, element_column_map, element_eref_map,
                                element_y2x_map, num_metal, epot, dimensionless=True)
                        phi_kn.append(phi.reshape(-1,1))

        if os.path.isfile(f'{self.calculation_dir}/{self.name}_mbar.traj'):
            images = read(f'{self.calculation_dir}/{self.name}_mbar.traj', index=':')
        else:
            write(f'{self.calculation_dir}/{self.name}_mbar.traj', images)

        phi_kn = np.hstack(phi_kn)  # Shape (K, N_total)
        # Build numpy array of which state produced each sample (global column index)
        sample_state = np.asarray(sample_state, dtype=int)  # shape (N_total,)
        sample_steps = np.asarray(sample_steps, dtype=int)  # shape (N_total,)
        N_total = sample_state.size
        kept_indices_list = []
        present_states, present_counts = np.unique(sample_state, return_counts=True)
        parprint(f"Found {present_states.size} occupied states (out of {K}). Subsampling each occupied state's series...")

        for s, n_k in zip(present_states, present_counts):
            idx_k = np.where(sample_state == s)[0]     # global indices of samples from state s (chronological)
            if n_k == 0:
                continue
            if sample_stride is not None:
                # Interpret stride as simulation-step bin width: use original simulation steps (sample_steps)
                stride_int = int(sample_stride)
                steps_k = sample_steps[idx_k]           # original n_minus_1 values for samples of this state
                bins = steps_k // stride_int
                # pick the first sample that appears in each bin (preserves chronology)
                #unique_bins, first_pos = np.unique(bins, return_index=True)
                #subs_global = idx_k[first_pos]           # convert to global indices
                # take last occurrence per bin instead of first
                unique_bins, last_pos_rev = np.unique(bins[::-1], return_index=True)
                last_pos = bins.size - 1 - last_pos_rev
                subs_global = idx_k[last_pos]
            else:
                # Let pymbar compute statistical inefficiency and return per-state local indices
                series = phi_kn[int(s), idx_k].ravel()
                try:
                    local_idx = np.array(timeseries.subsample_correlated_data(series))  # indices into series
                    subs_global = idx_k[local_idx]
                except Exception as e:
                    parprint(f'  warning: subsample_correlated_data failed for state {s}: {e}; keeping all samples for this state')
                    subs_global = idx_k.copy()
            if subs_global.size == 0:
                parprint(f'  state {s}: no subsampled indices -> skipping')
                continue
            kept_indices_list.append(subs_global)
            parprint(f' state {s}: n_k={n_k}, kept={subs_global.size}')

        if len(kept_indices_list) == 0:
            uniq, counts = np.unique(sample_state, return_counts=True)
            raise RuntimeError(f'No subsampled indices found for any occupied state. ' +
                               f'Present states: {uniq}, counts: {counts}. phi_kn.shape={phi_kn.shape}')
        kept_indices = np.unique(np.concatenate(kept_indices_list)).astype(int)
        kept_indices.sort()
        np.save(f'{self.calculation_dir}/{self.name}_sampled_indices.npy', kept_indices)
        parprint(f'Original total samples: {N_total}, subsampled total: {kept_indices.size}')

        # make subsampled phi_kn and per-state counts for MBAR
        phi_kn_sub = phi_kn[:, kept_indices]
        samples_per_state_sub = np.array([np.sum(sample_state[kept_indices] == k) for k in range(K)], dtype=int)
        #parprint(f'New per-state sample counts (subsampled): {samples_per_state_sub}')
        images_sub = [images[i] for i in kept_indices]

        fes = FES(phi_kn_sub, samples_per_state_sub,
                  mbar_options={'relative_tolerance': 1e-7, 'solver_protocol': 'robust', 'maximum_iterations': 30000})
        fes_map = {}
        for cv in variable_keys:
            if cv in ['fragments', 'x_surf_metals', 'ads_coverages', 'raw_scores', 'mu', 'mu_window']:
                continue
            if cv == 'all':
                x_n = np.array([np.concatenate([a.info['data']['x_surf_metals'], a.info['data']['ads_coverages']]) for a in images_sub])
                bins = [np.arange(0., 1.001, bin_width) for _ in range(x_n.shape[1])]
            else:
                x_n = np.array([a.info['data'][cv] for a in images_sub])
                bins = [np.arange(0., 1.001, bin_width)]

            free_energies = np.full(shape=(len(images), K), fill_value=np.nan)
            if sample_condition_indices is None:
                all_condition_indices = list(range(K))
            else:
                all_condition_indices = sample_condition_indices
            for k in sample_condition_indices:
                fes.generate_fes(phi_kn_sub[k,:], x_n, fes_type='histogram', histogram_parameters={'bin_edges': bins})
                results = fes.get_fes(x_n)
                f_n = results['f_i']
                dd = defaultdict(list)
                for i, f in enumerate(f_n):
                    dd[f].append(i)
                for f, indices in dd.items():
                    free_energies[kept_indices[indices],k] = f
            fes_map[cv] = free_energies

        with open(f'{self.calculation_dir}/{self.name}_mbar_fes.pkl', 'wb') as handle:
            pickle.dump(fes_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return fes_map
