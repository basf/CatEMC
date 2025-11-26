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
        temperature_key='temperature',
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
        fct1, fct2 = ect1, ect2
    else:
        fct1 = Counter(atoms.info['data']['fragments'])
        fct2 = Counter(trial_atoms.info['data']['fragments'])
        if no_internal_indices is not None:
            no_internal_indices1 = [orig_to_new[i] for i in no_internal_indices if i in orig_to_new]
            nfct1, nfct2 = fct1.copy(), fct2.copy()
            lfct1, lfct2 = {}, {} 
            lattice_specs1 = atoms.symbols[no_internal_indices1]
            for s in set(lattice_specs1):
                if s in nfct1:
                    ns = len([spec for spec in lattice_specs1 if spec == s])
                    nfct1[s] -= ns
                    lfct1[s] = ns
            no_internal_indices2 = [trial_to_new[i] for i in no_internal_indices if i in trial_to_new]
            lattice_specs2 = trial_atoms.symbols[no_internal_indices2]
            for s in set(lattice_specs2):
                if s in nfct2:
                    ns = len([spec for spec in lattice_specs2 if spec == s])
                    nfct2[s] -= ns
                    lfct2[s] = ns

    species = set(list(fct1.keys()) + list(fct2.keys()) + list(ect1.keys()) + list(ect2.keys()))
    prefactor = 1
    for k in species:
        if k in nonmetal_species:
            delta_Nk = fct2[k] - fct1[k]
            if delta_Nk == 0:
                try:
                    prefactor *= (math.factorial(fct1[k]) / math.factorial(fct2[k]))
                except:
                    prefactor *= np.exp(stirling(fct1[k]) - stirling(fct2[k]))
            else:
                if temperature_species_internal_map is None:
                    mass = data.atomic_masses[data.atomic_numbers[k]] * units._amu
                else:
                    ads_atoms = adsorbate_molecule(k)
                    mass = np.sum(ads_atoms.get_masses()) * units._amu 
                # thermal de broglie wavelength
                wavelength = hplanck / math.sqrt(2 * math.pi * mass * kbT_J) * 1e10 
                if temperature_species_internal_map is None:
                    frac = (volume / wavelength**3)**delta_Nk
                else:
                    if (no_internal_indices is not None) and (k in lfct1) and (k in lfct2):
                        frac = (volume * temperature_species_internal_map[T][k] / wavelength**3)**(
                                nfct2[k] - nfct1[k]) * (volume / wavelength**3)**(lfct2[k] - lfct1[k])
                    else:
                        frac = (volume * temperature_species_internal_map[T][k] / wavelength**3)**delta_Nk
                try:
                    prefactor *= (math.factorial(fct1[k]) / math.factorial(fct2[k]) * frac)
                except:
                    prefactor *= (np.exp(stirling(fct1[k]) - stirling(fct2[k])) * frac)
        else:
            m, n = None, None
            if ect2[k] - ect1[k] == 1:
                n = k
            if ect1[k] - ect2[k] == 1:
                m = k
            try:
                prefactor *= (math.factorial(ect1[k]) / math.factorial(ect2[k]))
            except:
                prefactor *= np.exp(stirling(ect1[k]) - stirling(ect2[k]))

    if (metal_species_probabilities is not None) and (m is not None) and (n is not None):
        prefactor *= (metal_species_probabilities[m] * (1 - metal_species_probabilities[m]))
        prefactor /= (metal_species_probabilities[n] * (1 - metal_species_probabilities[n]))

    return prefactor


def compute_interswap_prefactor(
        atoms1, atoms2, trial_atoms1, trial_atoms2=None):

    ect11, ect12 = Counter(atoms1.symbols), Counter(trial_atoms1.symbols)
    mutated_elements = [k for k, v in ect11.items() if v != ect12[k]]
    mutated_elements.sort(key=lambda x: ect12[x] - ect11[x])
    m1, m2 = mutated_elements
    N11 = len([a for a in atoms1 if a.symbol == m1])
    N12 = len([a for a in atoms1 if a.symbol == m2])
    N21 = len([a for a in atoms2 if a.symbol == m1])
    N22 = len([a for a in atoms2 if a.symbol == m2])
    prefactor = N11 * N22 / (N12 + 1) / (N21 + 1)

    return prefactor


def compute_crossover_prefactor(
        atoms1, atoms2, trial_atoms1, trial_atoms2, 
        nonmetal_species=None, 
        temperature_species_internal_map=None,
        no_internal_indices=None,
        temperature_key='temperature'):

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
        fct11, fct12, fct21, fct22 = ect11, ect12, ect21, ect22        
    else:
        ads11 = atoms2.info['data']['fragments']
        ads12 = trial_atoms2.info['data']['fragments']
        ads21 = atoms2.info['data']['fragments']
        ads22 = trial_atoms2.info['data']['fragments']
        fct11, fct12 = Counter(ads11), Counter(ads12)
        fct21, fct22 = Counter(ads21), Counter(ads22)
        if no_internal_indices is not None:
            no_internal_indices11 = [orig_to_new1[i] for i in no_internal_indices if i in orig_to_new1]
            lattice_specs11 = atoms1.symbols[no_internal_indices11]
            nfct11, nfct12, nfct21, nfct22 = fct11.copy(), fct12.copy(), fct21.copy(), fct22.copy()
            for s in set(lattice_specs11):
                if s in nfct11:
                    nfct11[s] -= len([spec for spec in lattice_specs11 if spec == s])
            no_internal_indices12 = [trial_to_new1[i] for i in no_internal_indices if i in trial_to_new1]
            lattice_specs12 = trial_atoms1.symbols[no_internal_indices12]
            for s in set(lattice_specs12):
                if s in nfct12:
                    nfct12[s] -= len([spec for spec in lattice_specs12 if spec == s])
            no_internal_indices21 = [orig_to_new2[i] for i in no_internal_indices if i in orig_to_new2]
            lattice_specs21 = atoms2.symbols[no_internal_indices21]
            for s in set(lattice_specs21):
                if s in nfct21:
                    nfct21[s] -= len([spec for spec in lattice_specs21 if spec == s])
            no_internal_indices22 = [trial_to_new2[i] for i in no_internal_indices if i in trial_to_new2]
            lattice_specs22 = trial_atoms2.symbols[no_internal_indices22]
            for s in set(lattice_specs22):
                if s in nfct22:
                    nfct22[s] -= len([spec for spec in lattice_specs22 if spec == s])

    prefactor = 1
    for k in set(list(fct11.keys()) + list(fct12.keys()) + list(fct21.keys()) + list(fct22.keys()) +
                 list(ect11.keys()) + list(ect12.keys()) + list(ect21.keys()) + list(ect22.keys())):
        if k in nonmetal_species:
            delta_Nk1 = fct12[k] - fct11[k]
            delta_Nk2 = fct22[k] - fct21[k]
            if (delta_Nk1 == 0) and (delta_Nk2 == 0):
                try:
                    prefactor *= (math.factorial(fct11[k]) / math.factorial(fct12[k]) * math.factorial(fct21[k]) 
                                  / math.factorial(fct22[k]))
                except:
                    prefactor *= np.exp(stirling(fct11[k]) - stirling(fct12[k]) + stirling(fct21[k]) - stirling(fct22[k]))
            else:
                if temperature_species_internal_map is None:
                    frac1 = beta1**(-1.5)
                    frac2 = beta2**(-1.5)
                else:
                    frac1 = temperature_species_internal_map[T1][k] / beta1**1.5
                    frac2 = temperature_species_internal_map[T2][k] / beta2**1.5
                    if no_internal_indices is not None:
                        delta_Nk1 = nfct12[k] - nfct11[k]
                        delta_NK2 = nfct22[k] - nfct21[k]
                try:
                    prefactor *= (math.factorial(fct11[k]) / math.factorial(fct12[k]) * math.factorial(fct21[k]) 
                                  / math.factorial(fct22[k]) * frac1**delta_Nk1 * frac2**delta_Nk2)
                except:
                    prefactor *= (np.exp(stirling(fct11[k]) - stirling(fct12[k]) + stirling(fct21[k]) - stirling(fct22[k]))
                                  * frac1**delta_Nk1 * frac2**delta_Nk2)
        else:
            try:
                prefactor *= (math.factorial(ect11[k]) / math.factorial(ect12[k]) * math.factorial(ect21[k]) 
                              / math.factorial(ect22[k]))
            except:
                prefactor *= np.exp(stirling(ect11[k]) - stirling(ect12[k]) + stirling(ect21[k]) - stirling(ect22[k]))

    return prefactor


def compute_exchange_prefactor(
        atoms1, atoms2, 
        reaction_conditions=None, 
        element_column_map=None, 
        temperature_species_internal_map=None,
        no_internal_indices=None,
        temperature_key='temperature',
        exponent=None,
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
        if element_column_map is None:
            N_frag1, N_frag2 = 0, 0
        else:
            elements = element_column_map.keys()
            N_frag1 = len([a for a in atoms1 if a.symbol in elements])
            N_frag2 = len([a for a in atoms2 if a.symbol in elements])
    else:
        N_frag1 = len(atoms1.info['data']['fragments'])       
        N_frag2 = len(atoms2.info['data']['fragments'])

    prefactor = (beta2/beta1)**(1.5*(N_frag2 - N_frag1))
    if temperature_species_internal_map is not None:
        fct1 = Counter(atoms1.info['data']['fragments'])
        fct2 = Counter(atoms2.info['data']['fragments'])
        if no_internal_indices is not None:
            no_internal_indices1 = [orig_to_new1[i] for i in no_internal_indices if i in orig_to_new1]
            lattice_specs1 = atoms1.symbols[no_internal_indices1]
            for s in set(lattice_specs1):
                if s in fct1:
                    fct1[s] -= len([spec for spec in lattice_specs1 if spec == s])
            no_internal_indices2 = [orig_to_new2[i] for i in no_internal_indices if i in orig_to_new2]
            lattice_specs2 = atoms2.symbols[no_internal_indices2]
            for s in set(lattice_specs2):
                if s in fct2:
                    fct2[s] -= len([spec for spec in lattice_specs2 if spec == s])
        prefactor *= np.prod([(temperature_species_internal_map[T1][k] / 
                temperature_species_internal_map[T2][k])**(fct2[k] - fct1[k]) 
                for k in temperature_species_internal_map[T1].keys()])

    if exponent is None:
        exponent = 0
        if element_column_map is not None:
            for k in set(list(ect1.keys()) + list(ect2.keys())):
                if k in element_column_map:
                    col = element_column_map[k]
                    if reaction_conditions is None:
                        mu1 = atoms1.info['data']['mu'][col]
                        mu2 = atoms2.info['data']['mu'][col]
                    else:
                        mu1 = reaction_conditions[:, col]
                        mu2 = reaction_conditions[:, col]
                    exponent += (beta1*ect2[k] - beta2*ect1[k]) * (mu2 - mu1)
    prefactor /= np.exp(exponent)

    return prefactor


def stirling(x):

    return x * (np.log(x) - 1) + np.log((2*np.pi*x)**0.5)


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
                # fill dmu_T vector using provided temperature_dmu_map
                for T_key, dmap in temperature_dmu_map.items():
                    if pair_key in dmap:
                        dmu_T[T_arr==T_key] = dmap[pair_key]
                metal_mu_sum += (dc_surf - dc_bulk) * dmu_T
    phi = epot - num_metal / len(target_metal_species) * metal_mu_sum

    # non-metal contributions
    if element_column_map:
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
