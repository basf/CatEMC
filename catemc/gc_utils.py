#!/usr/bin/env python
# -*- coding: utf-8 -*-
from acat.settings import (adsorbate_elements, 
                           adsorbate_molecule)
from ase import Atoms, data, units
from itertools import combinations
from collections import Counter
import numpy as np
import math


def compute_intramove_prefactor(
        atoms, trial_atoms, volume,
        nonmetal_species=None,
        temperature_species_internal_map=None,
        no_internal_indices=None,
        metal_species_probabilities=None,
        temperature_key='temperature', return_log=False,
        ):

    real_indices = [a.index for a in atoms if a.symbol != 'X']
    atoms = atoms[real_indices]
    orig_to_new = {idx: i for i, idx in enumerate(real_indices)}
    trial_real_indices = [a.index for a in trial_atoms if a.symbol != 'X']
    trial_atoms = trial_atoms[trial_real_indices]
    trial_to_new = {idx: i for i, idx in enumerate(trial_real_indices)}
    T = atoms.info['key_value_pairs'][temperature_key]
    kBT_eV = units.kB * T
    beta = 1. / kBT_eV  # 1/(kb*T), eV
    hplanck = units._hplanck  # J/Hz = kg*m2*s-1
    kbT_J = kBT_eV * units._e  # J = kg*m2*s-2
    if nonmetal_species is None:
        nonmetal_species = adsorbate_elements
    ect1, ect2 = Counter(atoms.symbols), Counter(trial_atoms.symbols)
    if temperature_species_internal_map is None:
        fct1, fct2 = ect1.copy(), ect2.copy()
    else:
        fct1 = Counter(atoms.info['data']['fragments'])
        fct2 = Counter(trial_atoms.info['data']['fragments'])
        nonmetal_species = set(nonmetal_species) | set(temperature_species_internal_map[T])
    if no_internal_indices is not None:
        no_internal_indices1 = [orig_to_new[i] for i in no_internal_indices if i in orig_to_new]
        lattice_specs1 = atoms.symbols[no_internal_indices1]
        for k in set(lattice_specs1):
            if k in fct1:
                fct1[k] -= list(lattice_specs1).count(k)
        no_internal_indices2 = [trial_to_new[i] for i in no_internal_indices if i in trial_to_new]
        lattice_specs2 = trial_atoms.symbols[no_internal_indices2]
        for k in set(lattice_specs2):
            if k in fct2:
                fct2[k] -= list(lattice_specs2).count(k)
    if any(n < 0 for c in (fct1, fct2) for n in c.values()):
        raise ValueError('Fragment counts are inconsistent with the lattice indices.')

    species = set(list(fct1.keys()) + list(fct2.keys()) + list(ect1.keys()) + list(ect2.keys()))
    proposal_log = trial_atoms.info.get('data', {}).get('mc_log_proposal_ratio')
    ln_prefactor = 0. if proposal_log is None else float(proposal_log)
    m, n = None, None
    for k in species:
        if k in nonmetal_species:
            delta_Nk = fct2[k] - fct1[k]
            ln_prefactor += stirling(fct1[k]) - stirling(fct2[k])
            if delta_Nk != 0:
                if volume is None or not np.isfinite(volume) or volume <= 0:
                    raise ValueError('A positive accessible volume is required for mobile-fragment changes.')
                if temperature_species_internal_map is None:
                    mass = data.atomic_masses[data.atomic_numbers[k]] * units._amu
                    Q = 1.
                else:
                    ads_atoms = adsorbate_molecule(k)
                    mass = np.sum(ads_atoms.get_masses()) * units._amu
                    Q = temperature_species_internal_map[T][k]
                if Q <= 0 or not np.isfinite(Q):
                    raise ValueError('Internal partition functions must be finite and positive.')
                # thermal de broglie wavelength
                wavelength = hplanck / math.sqrt(2 * math.pi * mass * kbT_J) * 1e10
                ln_prefactor += delta_Nk * (np.log(volume) + np.log(Q) - 3*np.log(wavelength))
        else:
            if ect2[k] - ect1[k] == 1:
                n = k
            if ect1[k] - ect2[k] == 1:
                m = k
            # Explicit metal-site assignments already include mixing multiplicity.

    if (proposal_log is None and metal_species_probabilities is not None
            and m is not None and n is not None):
        W = sum(metal_species_probabilities.values())
        wm, wn = metal_species_probabilities[m], metal_species_probabilities[n]
        ln_prefactor += np.log(wm) + np.log(W-wm) - np.log(wn) - np.log(W-wn)

    return ln_prefactor if return_log else np.exp(ln_prefactor)


def compute_interswap_prefactor(
        atoms1, atoms2, trial_atoms1, trial_atoms2=None):

    proposal_log = trial_atoms1.info.get('data', {}).get('mc_log_proposal_ratio')
    if proposal_log is None:
        raise ValueError('Interswap requires the proposal ratio recorded by EMCInterSwap.')
    return np.exp(proposal_log)



def compute_crossover_prefactor(
        atoms1, atoms2, trial_atoms1, trial_atoms2, 
        nonmetal_species=None, 
        temperature_species_internal_map=None,
        no_internal_indices=None,
        temperature_key='temperature', volume=None, return_log=False):

    real_indices1 = [a.index for a in atoms1 if a.symbol != 'X']
    atoms1 = atoms1[real_indices1]
    orig_to_new1 = {idx: i for i, idx in enumerate(real_indices1)}
    real_indices2 = [a.index for a in atoms2 if a.symbol != 'X']
    atoms2 = atoms2[real_indices2]
    orig_to_new2 = {idx: i for i, idx in enumerate(real_indices2)}
    trial_real_indices1 = [a.index for a in trial_atoms1 if a.symbol != 'X']
    trial_atoms1 = trial_atoms1[trial_real_indices1]
    trial_to_new1 = {idx: i for i, idx in enumerate(trial_real_indices1)}
    trial_real_indices2 = [a.index for a in trial_atoms2 if a.symbol != 'X']
    trial_atoms2 = trial_atoms2[trial_real_indices2]
    trial_to_new2 = {idx: i for i, idx in enumerate(trial_real_indices2)}
    T1 = atoms1.info['key_value_pairs'][temperature_key]
    kBT1_eV = units.kB * T1
    beta1 = 1. / kBT1_eV  # 1/(kb*T), eV
    T2 = atoms2.info['key_value_pairs'][temperature_key]
    kBT2_eV = units.kB * T2
    beta2 = 1. / kBT2_eV  # 1/(kb*T), eV
    if nonmetal_species is None:
        nonmetal_species = adsorbate_elements
    ect11, ect12 = Counter(atoms1.symbols), Counter(trial_atoms1.symbols)
    ect21, ect22 = Counter(atoms2.symbols), Counter(trial_atoms2.symbols)
    if temperature_species_internal_map is None:
        fct11, fct12 = ect11.copy(), ect12.copy()
        fct21, fct22 = ect21.copy(), ect22.copy()
    else:
        ads11 = atoms1.info['data']['fragments']
        ads12 = trial_atoms1.info['data']['fragments']
        ads21 = atoms2.info['data']['fragments']
        ads22 = trial_atoms2.info['data']['fragments']
        fct11, fct12 = Counter(ads11), Counter(ads12)
        fct21, fct22 = Counter(ads21), Counter(ads22)
        nonmetal_species = set(nonmetal_species) | set(temperature_species_internal_map[T1])
    if no_internal_indices is not None:
        no_internal_indices11 = [orig_to_new1[i] for i in no_internal_indices if i in orig_to_new1]
        lattice_specs11 = atoms1.symbols[no_internal_indices11]
        for k in set(lattice_specs11):
            if k in fct11:
                fct11[k] -= list(lattice_specs11).count(k)
        no_internal_indices12 = [trial_to_new1[i] for i in no_internal_indices if i in trial_to_new1]
        lattice_specs12 = trial_atoms1.symbols[no_internal_indices12]
        for k in set(lattice_specs12):
            if k in fct12:
                fct12[k] -= list(lattice_specs12).count(k)
        no_internal_indices21 = [orig_to_new2[i] for i in no_internal_indices if i in orig_to_new2]
        lattice_specs21 = atoms2.symbols[no_internal_indices21]
        for k in set(lattice_specs21):
            if k in fct21:
                fct21[k] -= list(lattice_specs21).count(k)
        no_internal_indices22 = [trial_to_new2[i] for i in no_internal_indices if i in trial_to_new2]
        lattice_specs22 = trial_atoms2.symbols[no_internal_indices22]
        for k in set(lattice_specs22):
            if k in fct22:
                fct22[k] -= list(lattice_specs22).count(k)
    if any(n < 0 for c in (fct11, fct12, fct21, fct22) for n in c.values()):
        raise ValueError('Fragment counts are inconsistent with the lattice indices.')

    ln_prefactor = 0.
    for k in set(list(fct11.keys()) + list(fct12.keys()) + list(fct21.keys()) + list(fct22.keys())):
        if k in nonmetal_species:
            delta_Nk1 = fct12[k] - fct11[k]
            delta_Nk2 = fct22[k] - fct21[k]
            ln_prefactor += stirling(fct11[k]) - stirling(fct12[k]) + stirling(fct21[k]) - stirling(fct22[k])
            if delta_Nk1 != 0 or delta_Nk2 != 0:
                ln_frac1, ln_frac2 = -1.5*np.log(beta1), -1.5*np.log(beta2)
                if temperature_species_internal_map is not None:
                    Q1, Q2 = temperature_species_internal_map[T1][k], temperature_species_internal_map[T2][k]
                    if min(Q1, Q2) <= 0 or not np.isfinite(Q1 + Q2):
                        raise ValueError('Internal partition functions must be finite and positive.')
                    ln_frac1 += np.log(Q1)
                    ln_frac2 += np.log(Q2)
                ln_prefactor += delta_Nk1*ln_frac1 + delta_Nk2*ln_frac2
                if delta_Nk1 + delta_Nk2 != 0:
                    if volume is None or not np.isfinite(volume) or volume <= 0:
                        raise ValueError('Fragment-changing crossover requires a positive accessible volume.')
                    if temperature_species_internal_map is None:
                        mass = data.atomic_masses[data.atomic_numbers[k]] * units._amu
                    else:
                        mass = sum(adsorbate_molecule(k).get_masses()) * units._amu
                    # beta is in 1/eV. This is the wavelength at kB*T = 1 eV.
                    wavelength = units._hplanck / math.sqrt(2*math.pi*mass*units._e) * 1e10
                    ln_prefactor += (delta_Nk1 + delta_Nk2) * (np.log(volume) - 3*np.log(wavelength))

    return ln_prefactor if return_log else np.exp(ln_prefactor)


def compute_exchange_prefactor(
        atoms1, atoms2, 
        element_column_map=None, 
        temperature_species_internal_map=None,
        no_internal_indices=None,
        temperature_key='temperature',
        exponent=None, return_log=False,
        ):

    real_indices1 = [a.index for a in atoms1 if a.symbol != 'X']
    atoms1 = atoms1[real_indices1]
    orig_to_new1 = {idx: i for i, idx in enumerate(real_indices1)}
    real_indices2 = [a.index for a in atoms2 if a.symbol != 'X']
    atoms2 = atoms2[real_indices2]
    orig_to_new2 = {idx: i for i, idx in enumerate(real_indices2)}
    T1 = atoms1.info['key_value_pairs'][temperature_key]
    kBT1_eV = units.kB * T1
    beta1 = 1. / kBT1_eV  # 1/(kb*T), eV
    T2 = atoms2.info['key_value_pairs'][temperature_key]
    kBT2_eV = units.kB * T2
    beta2 = 1. / kBT2_eV  # 1/(kb*T), eV
    ect1, ect2 = Counter(atoms1.symbols), Counter(atoms2.symbols)
    if temperature_species_internal_map is None:
        elements = () if element_column_map is None else element_column_map.keys()
        fct1 = Counter(a.symbol for a in atoms1 if a.symbol in elements)
        fct2 = Counter(a.symbol for a in atoms2 if a.symbol in elements)
    else:
        fct1 = Counter(atoms1.info['data']['fragments'])
        fct2 = Counter(atoms2.info['data']['fragments'])
    if no_internal_indices is not None:
        no_internal_indices1 = [orig_to_new1[i] for i in no_internal_indices if i in orig_to_new1]
        lattice_specs1 = atoms1.symbols[no_internal_indices1]
        for k in set(lattice_specs1):
            if k in fct1:
                fct1[k] -= list(lattice_specs1).count(k)
        no_internal_indices2 = [orig_to_new2[i] for i in no_internal_indices if i in orig_to_new2]
        lattice_specs2 = atoms2.symbols[no_internal_indices2]
        for k in set(lattice_specs2):
            if k in fct2:
                fct2[k] -= list(lattice_specs2).count(k)
    if any(n < 0 for c in (fct1, fct2) for n in c.values()):
        raise ValueError('Fragment counts are inconsistent with the lattice indices.')
    N_frag1, N_frag2 = sum(fct1.values()), sum(fct2.values())
    ln_prefactor = 1.5*(N_frag2 - N_frag1)*np.log(beta2/beta1)
    if temperature_species_internal_map is not None:
        for k in fct1.keys() | fct2.keys():
            if fct2[k] != fct1[k]:
                Q1, Q2 = temperature_species_internal_map[T1][k], temperature_species_internal_map[T2][k]
                if min(Q1, Q2) <= 0 or not np.isfinite(Q1 + Q2):
                    raise ValueError('Internal partition functions must be finite and positive.')
                ln_prefactor += (fct2[k] - fct1[k]) * (np.log(Q1) - np.log(Q2))

    if exponent is None:
        exponent = 0
        if element_column_map is not None:
            for k in set(list(ect1.keys()) + list(ect2.keys())):
                if k in element_column_map:
                    col = element_column_map[k]
                    mu1 = atoms1.info['data']['mu'][col]
                    mu2 = atoms2.info['data']['mu'][col]
                    exponent += (beta1*ect2[k] - beta2*ect1[k]) * (mu2 - mu1)
    ln_prefactor -= exponent

    return ln_prefactor if return_log else np.exp(ln_prefactor)


def stirling(x):

    return math.lgamma(x + 1)


def compute_phi(num_dict, target_metal_species, target_metal_composition,
                temperature_dmu_map, grid, element_column_map, element_eref_map, 
                element_y2x_map, num_metal, epot, dimensionless=True):
    """
    Vectorized function for computing microscopic GC potential energies.
    - accepts grid as either 1D array of temperatures or 2D array where last column is T;
    - if temperature_dmu_map is None, metal contributions are treated as zero;
    - if element_column_map is None or empty, non-metal contributions are treated as zero.
    """

    # normalize bulk composition
    c_sum = sum(target_metal_composition.values())
    if c_sum != 1.:
        target_metal_composition = {k: v / c_sum for k, v in target_metal_composition.items()}
    grid = np.asarray(grid)
    if grid.ndim == 1:
        T_arr = grid
    elif grid.ndim == 2:
        T_arr = grid[:, -1]
    else:
        raise ValueError('compute_phi: unexpected grid shape')

    # metal chemical potential contribution
    if temperature_dmu_map is None:
        metal_mu_sum = np.zeros_like(T_arr, dtype=float)
    else:
        # temperature_dmu_map expected: {T: {'A2B': dmu_AB, ... }, ...}
        metal_mu_sum = np.zeros_like(T_arr, dtype=float)
        for m in target_metal_species:
            for n in target_metal_species:
                if m == n:
                    continue
                dc_surf = (num_dict.get(n, 0.) - num_dict.get(m, 0.)) / num_metal 
                dc_bulk = target_metal_composition.get(n, 0.) - target_metal_composition.get(m, 0.)
                pair_key = f"{m}2{n}"
                dmu_T = np.zeros_like(T_arr, dtype=float)
                for T in np.unique(T_arr):
                    if T not in temperature_dmu_map:
                        raise ValueError(f'Metal chemical potentials are missing at T={T}.')
                    dmap = temperature_dmu_map[T]
                    if pair_key not in dmap and f'{n}2{m}' not in dmap:
                        raise ValueError(f'Metal chemical potential difference {m}/{n} is missing at T={T}.')
                # fill dmu_T vector using provided temperature_dmu_map
                for T_key, dmap in temperature_dmu_map.items():
                    if pair_key in dmap:
                        value = dmap[pair_key]
                        if not np.isfinite(value):
                            raise ValueError('Metal chemical potential differences must be finite.')
                        if f'{n}2{m}' in dmap:
                            if not np.isclose(value, -dmap[f'{n}2{m}'], rtol=1e-10, atol=1e-10):
                                raise ValueError('Metal chemical potential differences must be antisymmetric.')
                            value *= 0.5  # both directions together count the pair once
                        dmu_T[T_arr==T_key] = value
                metal_mu_sum += (dc_surf - dc_bulk) * dmu_T
    phi = epot - num_metal / len(target_metal_species) * metal_mu_sum

    # non-metal contributions
    if element_column_map is not None:
        for elem, col in element_column_map.items():
            nk = num_dict.get(elem, 0)
            y2x = (element_y2x_map.get(elem, 0.) if element_y2x_map is not None else 0.)
            eref = (element_eref_map.get(elem, 0.) if element_eref_map is not None else 0.)
            # grid handling: if grid 2D, assume chemical potential is in grid[:,col]; if grid 1D,
            # then user must have passed the correct per-state values as a 1D array aligned with T
            if grid.ndim == 2:
                mu_vals = grid[:, col]
            else:
                # grid is 1D (temperature-only): there are no mu dims -> treat mu as zero
                mu_vals = np.zeros_like(T_arr, dtype=float)
            phi -= (nk - y2x * num_metal) * (mu_vals + eref)

    if dimensionless:
        phi /= (units.kB * T_arr)

    return phi
