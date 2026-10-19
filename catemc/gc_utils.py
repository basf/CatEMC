#!/usr/bin/env python
# -*- coding: utf-8 -*-
from acat.settings import adsorbate_elements, adsorbate_molecule
from ase import data, units
from itertools import combinations
from collections import Counter
import numpy as np
import math


def _mobile_fragment_counts(atoms, nonmetal_species=None,
                            temperature_species_internal_map=None,
                            no_internal_indices=None):
    """Count mobile fragments, excluding explicit catalyst-site occupations."""
    if temperature_species_internal_map is None:
        species = set(adsorbate_elements if nonmetal_species is None else nonmetal_species)
        counts = Counter(a.symbol for a in atoms if a.symbol != 'X' and a.symbol in species)
    else:
        counts = Counter(atoms.info['data']['fragments'])
    for i in set(() if no_internal_indices is None else no_internal_indices):
        if 0 <= i < len(atoms) and atoms[i].symbol in counts:
            counts[atoms[i].symbol] -= 1
    if any(n < 0 for n in counts.values()):
        raise ValueError('Fragment counts are inconsistent with the lattice indices.')
    return +counts


def compute_intramove_prefactor(
        atoms, trial_atoms, volume, nonmetal_species=None,
        temperature_species_internal_map=None, no_internal_indices=None,
        metal_species_probabilities=None, temperature_key='temperature',
        return_log=False):
    proposal_log = trial_atoms.info.get('data', {}).get('mc_log_proposal_ratio')
    fixed = set(() if no_internal_indices is None else no_internal_indices)
    if proposal_log is not None:
        # The mutated indices are discrete occupations even for standalone use.
        if len(atoms) == len(trial_atoms):
            fixed.update(np.flatnonzero(atoms.numbers != trial_atoms.numbers))
        log_ratio = float(proposal_log)
    else:
        log_ratio = 0.
    before = _mobile_fragment_counts(atoms, nonmetal_species,
                                     temperature_species_internal_map, fixed)
    after = _mobile_fragment_counts(trial_atoms, nonmetal_species,
                                    temperature_species_internal_map, fixed)
    T = atoms.info['key_value_pairs'][temperature_key]
    for k in before.keys() | after.keys():
        n0, n1 = before[k], after[k]
        log_ratio += math.lgamma(n0 + 1) - math.lgamma(n1 + 1)
        if n1 != n0:
            if volume is None or not np.isfinite(volume) or volume <= 0:
                raise ValueError('A positive accessible volume is required for mobile-fragment changes.')
            if temperature_species_internal_map is None:
                mass = data.atomic_masses[data.atomic_numbers[k]] * units._amu
                Q = 1.
            else:
                mass = sum(adsorbate_molecule(k).get_masses()) * units._amu
                Q = temperature_species_internal_map[T][k]
            if Q <= 0 or not np.isfinite(Q):
                raise ValueError('Internal partition functions must be finite and positive.')
            wavelength = units._hplanck / math.sqrt(2*math.pi*mass*units.kB*T*units._e) * 1e10
            log_ratio += (n1-n0) * (np.log(volume) + np.log(Q) - 3*np.log(wavelength))
    if proposal_log is None and metal_species_probabilities is not None:
        a, b = Counter(atoms.symbols), Counter(trial_atoms.symbols)
        removed = [k for k in metal_species_probabilities if a[k]-b[k] == 1]
        added = [k for k in metal_species_probabilities if b[k]-a[k] == 1]
        if len(removed) == len(added) == 1:
            total = sum(metal_species_probabilities.values())
            wm, wn = metal_species_probabilities[removed[0]], metal_species_probabilities[added[0]]
            log_ratio += np.log(wm) + np.log(total-wm) - np.log(wn) - np.log(total-wn)
    return log_ratio if return_log else np.exp(log_ratio)


def compute_interswap_prefactor(atoms1, atoms2, trial_atoms1, trial_atoms2=None):
    """Reverse/forward unlike-index-pair proposal ratio, without target weights."""
    if 'mc_log_proposal_ratio' in trial_atoms1.info.get('data', {}):
        return np.exp(trial_atoms1.info['data']['mc_log_proposal_ratio'])
    if trial_atoms2 is None:
        raise ValueError('Both trial replicas are needed for an interswap ratio.')
    def unlike_pairs(a, b):
        ca, cb = Counter(a.symbols), Counter(b.symbols)
        return len(a)*len(b) - sum(v*cb[k] for k, v in ca.items())
    forward, reverse = unlike_pairs(atoms1, atoms2), unlike_pairs(trial_atoms1, trial_atoms2)
    return forward / reverse if reverse else 0.


def compute_crossover_prefactor(
        atoms1, atoms2, trial_atoms1, trial_atoms2, nonmetal_species=None,
        temperature_species_internal_map=None, no_internal_indices=None,
        temperature_key='temperature', volume=None, return_log=False):
    """Mobile-fragment target-weight ratio for a two-replica structural move.

    No proposal ratio is included. Volume/mass factors cancel only when each
    fragment type is conserved across the two replicas.
    """
    counts = [_mobile_fragment_counts(a, nonmetal_species,
                                     temperature_species_internal_map, no_internal_indices)
              for a in (atoms1, trial_atoms1, atoms2, trial_atoms2)]
    a, x, b, y = counts
    Ta = atoms1.info['key_value_pairs'][temperature_key]
    Tb = atoms2.info['key_value_pairs'][temperature_key]
    log_ratio = 0.
    for k in set().union(*(c.keys() for c in counts)):
        da, db = x[k]-a[k], y[k]-b[k]
        log_ratio += (math.lgamma(a[k]+1) + math.lgamma(b[k]+1)
                      - math.lgamma(x[k]+1) - math.lgamma(y[k]+1))
        log_ratio += 1.5*(da*np.log(Ta) + db*np.log(Tb))
        if temperature_species_internal_map is not None:
            Qa = temperature_species_internal_map[Ta][k]
            Qb = temperature_species_internal_map[Tb][k]
            if min(Qa, Qb) <= 0 or not np.isfinite(Qa + Qb):
                raise ValueError('Internal partition functions must be finite and positive.')
            log_ratio += da*np.log(Qa) + db*np.log(Qb)
        if da + db:
            if volume is None or not np.isfinite(volume) or volume <= 0:
                raise ValueError('Fragment-changing crossover requires a positive accessible volume.')
            if temperature_species_internal_map is None:
                mass = data.atomic_masses[data.atomic_numbers[k]] * units._amu
            else:
                mass = sum(adsorbate_molecule(k).get_masses()) * units._amu
            wavelength_at_1K = units._hplanck / math.sqrt(2*math.pi*mass*units.kB*units._e) * 1e10
            log_ratio += (da+db)*(np.log(volume) - 3*np.log(wavelength_at_1K))
    # Explicit metal-site assignments have no further metal count factorial.
    return log_ratio if return_log else np.exp(log_ratio)


def compute_exchange_prefactor(
        atoms1, atoms2, element_column_map=None,
        temperature_species_internal_map=None, no_internal_indices=None,
        temperature_key='temperature', exponent=None, return_log=False):
    elements = () if element_column_map is None else element_column_map.keys()
    a = _mobile_fragment_counts(atoms1, elements,
                               temperature_species_internal_map, no_internal_indices)
    b = _mobile_fragment_counts(atoms2, elements,
                               temperature_species_internal_map, no_internal_indices)
    Ta = atoms1.info['key_value_pairs'][temperature_key]
    Tb = atoms2.info['key_value_pairs'][temperature_key]
    ba, bb = 1/(units.kB*Ta), 1/(units.kB*Tb)
    log_ratio = 1.5*(sum(b.values())-sum(a.values()))*np.log(Ta/Tb)
    if temperature_species_internal_map is not None:
        for k in a.keys() | b.keys():
            Qa, Qb = temperature_species_internal_map[Ta][k], temperature_species_internal_map[Tb][k]
            if min(Qa, Qb) <= 0 or not np.isfinite(Qa + Qb):
                raise ValueError('Internal partition functions must be finite and positive.')
            log_ratio += (b[k]-a[k])*(np.log(Qa)-np.log(Qb))
    if exponent is None:
        exponent = 0.
        na, nb = Counter(atoms1.symbols), Counter(atoms2.symbols)
        for k, col in (element_column_map or {}).items():
            dmu = atoms2.info['data']['mu'][col] - atoms1.info['data']['mu'][col]
            exponent += (ba*nb[k] - bb*na[k])*dmu
    log_ratio -= exponent
    return log_ratio if return_log else np.exp(log_ratio)


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
        for m, n in combinations(target_metal_species, 2):
            dc_surf = (num_dict.get(n, 0.) - num_dict.get(m, 0.)) / num_metal
            dc_bulk = target_metal_composition.get(n, 0.) - target_metal_composition.get(m, 0.)
            dmu_T = np.empty_like(T_arr, dtype=float)
            for T in np.unique(T_arr):
                if T not in temperature_dmu_map:
                    raise ValueError(f'Metal chemical potentials are missing at T={T}.')
                dmap = temperature_dmu_map[T]
                forward, reverse = dmap.get(f'{m}2{n}'), dmap.get(f'{n}2{m}')
                if forward is None and reverse is None:
                    raise ValueError(f'Metal chemical potential difference {m}/{n} is missing at T={T}.')
                if forward is not None and reverse is not None and not np.isclose(
                        forward, -reverse, rtol=1e-10, atol=1e-10):
                    raise ValueError(f'Metal chemical potential differences {m}/{n} are not antisymmetric.')
                value = forward if forward is not None else -reverse
                if not np.isfinite(value):
                    raise ValueError('Metal chemical potential differences must be finite.')
                dmu_T[T_arr == T] = value
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
