#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mace.calculators.mace import MACECalculator
from ase.calculators.calculator import all_changes
import numpy as np


class MACECalculatorWrapper(MACECalculator):
    """
    MACECalculator wrapper that supports fixing atoms by zeroing forces,
    Hessian rows/cols, and per-atom dielectric derivatives for the given indices.

    Parameters
    ----------
    fix_indices : sequence[int], default None 
        Indices of atoms to fix. Accepts negative indices, boolean mask, or slice.
    All other args/kwargs are passed to MACECalculator.__init__.
    """

    def __init__(self, fix_indices=None, *args, device='cpu', 
                 default_dtype='', enable_cueq=False, **kwargs):

        kwargs_for_super = kwargs.copy()
        self.device = device
        kwargs_for_super['device'] = device
        self.default_dtype = default_dtype
        kwargs_for_super['default_dtype'] = default_dtype
        self.enable_cueq = enable_cueq
        kwargs_for_super['enable_cueq'] = enable_cueq
 
        super().__init__(*args, **kwargs_for_super)
        self._fix_indices = None
        if fix_indices is not None:
            self.set_fix_indices(fix_indices)

    def set_fix_indices(self, fix_indices, n_atoms=None):
        """
        Store/validate fix_indices. If n_atoms is provided, slices and boolean masks
        will be resolved immediately against that length.
        """
        if fix_indices is None:
            self._fix_indices = None
            return

        # boolean mask -> convert to indices
        if isinstance(fix_indices, (list, tuple, np.ndarray)) and len(fix_indices) > 0 and isinstance(fix_indices[0], (bool, np.bool_)):
            mask = np.asarray(fix_indices, dtype=bool)
            if n_atoms is not None and mask.shape[0] != n_atoms:
                raise ValueError("Boolean mask length does not match number of atoms.")
            self._fix_indices = np.nonzero(mask)[0].astype(int)
            return

        # slice -> resolve if n_atoms known, else store slice for later
        if isinstance(fix_indices, slice):
            if n_atoms is None:
                self._fix_indices = fix_indices
                return
            rng = range(*fix_indices.indices(n_atoms))
            self._fix_indices = np.fromiter(rng, dtype=int)
            return

        # otherwise assume sequence of ints
        idx_arr = np.asarray(fix_indices, dtype=int)
        if idx_arr.ndim != 1:
            raise ValueError("fix_indices must be a 1D sequence of ints, a boolean mask, or a slice.")
        self._fix_indices = idx_arr

    def _resolve_fix_indices_for_atoms(self, atoms):
        """Return validated, sorted, unique positive integer indices for this atoms object."""
        if self._fix_indices is None:
            return None

        n = len(atoms)

        # stored slice case: resolve now
        if isinstance(self._fix_indices, slice):
            rng = range(*self._fix_indices.indices(n))
            arr = np.fromiter(rng, dtype=int)
            return arr if arr.size > 0 else None

        arr = np.asarray(self._fix_indices, dtype=int, copy=True)
        # allow negative indexing
        arr = np.array([idx + n if idx < 0 else idx for idx in arr], dtype=int)

        if np.any(arr < 0) or np.any(arr >= n):
            raise IndexError(f"fix_indices contains out-of-range index for atoms with length {n}")

        arr = np.unique(arr)
        return arr if arr.size > 0 else None

    def _zero_per_atom_tensor(self, data, fix_idx):
        """
        Zero per-atom arrays/structures for fixed atom indices.

        Supports:
         - numpy.ndarray of shape (natoms, 3)
         - numpy.ndarray of shape (nmodels, natoms, 3)
         - list of numpy arrays following above shapes
        Returns modified object (same type).
        """
        if data is None:
            return data

        if isinstance(data, list):
            return [self._zero_per_atom_tensor(d, fix_idx) for d in data]

        if not isinstance(data, np.ndarray):
            # unexpected type: leave as-is
            return data

        if data.ndim == 3:
            # (nmodels, natoms, 3) -> zero [:, fix_idx, :]
            try:
                data[:, fix_idx, :] = 0.0
            except Exception:
                pass
            return data

        if data.ndim == 2 and data.shape[1] == 3 and data.shape[0] == len(fix_idx) + (data.shape[0] - len(fix_idx)):
            # ambiguous: prefer natoms x 3
            if data.shape[0] >= (fix_idx.max() + 1):
                data[fix_idx, :] = 0.0
            return data

        # other shapes (e.g., (nmodels, 3) or (natoms, k != 3)) -> try natoms on axis 0
        if data.ndim >= 2 and data.shape[0] >= (fix_idx.max() + 1):
            try:
                data[fix_idx, ...] = 0.0
            except Exception:
                pass
            return data

        return data

    def _zero_hessian_blocks(self, hess, fix_idx):
        """
        Given a 3N x 3N Hessian (numpy array), zero rows and columns
        corresponding to the fixed atom indices. Modifies and returns hess.
        """
        if hess is None:
            return hess
        if not isinstance(hess, np.ndarray):
            return hess
        # expected square
        if hess.ndim != 2 or hess.shape[0] != hess.shape[1]:
            return hess

        n3 = hess.shape[0]
        if n3 % 3 != 0:
            return hess
        n_atoms = n3 // 3

        # build index block list
        rows = []
        for idx in fix_idx:
            base = idx * 3
            rows.extend([base, base + 1, base + 2])
        rows = np.array(rows, dtype=int)

        # zero rows and columns
        hess[rows, :] = 0.0
        hess[:, rows] = 0.0
        return hess

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Call parent calculate, then zero forces for fixed indices.
        """
        # default signature compatibility
        if system_changes is None:
            from ase.calculators.calculator import all_changes
            system_changes = all_changes

        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        fix_idx = None
        if self._fix_indices is None:
            return

        if atoms is None and self.atoms is None:
            # nothing to resolve against
            return

        if atoms is None:
            atoms = self.atoms

        fix_idx = self._resolve_fix_indices_for_atoms(atoms)
        if fix_idx is None or len(fix_idx) == 0:
            return

        # Zero main forces array if present
        if "forces" in self.results and self.results["forces"] is not None:
            forces = self.results["forces"]
            try:
                forces[fix_idx, :] = 0.0
                self.results["forces"] = forces
            except Exception:
                # try other shapes / skip silently
                pass

        # If committee forces exist (forces_comm): shape (nmodels, natoms, 3)
        if "forces_comm" in self.results and self.results["forces_comm"] is not None:
            fcomm = self.results["forces_comm"]
            try:
                fcomm[:, fix_idx, :] = 0.0
                self.results["forces_comm"] = fcomm
            except Exception:
                pass

    def get_hessian(self, atoms=None):
        """
        Call parent get_hessian(), then zero Hessian rows/cols for fixed atoms.

        Returns the same type/form as the parent method: either a single ndarray
        (3N x 3N) or a list of such arrays (committee).
        """
        # delegate to parent
        hess = super().get_hessian(atoms=atoms)

        if self._fix_indices is None:
            return hess

        if atoms is None and self.atoms is None:
            return hess
        if atoms is None:
            atoms = self.atoms

        fix_idx = self._resolve_fix_indices_for_atoms(atoms)
        if fix_idx is None or len(fix_idx) == 0:
            return hess

        # If committee: list-like
        if isinstance(hess, list):
            return [self._zero_hessian_blocks(h.copy(), fix_idx) for h in hess]

        # Single numpy array
        if isinstance(hess, np.ndarray):
            return self._zero_hessian_blocks(hess.copy(), fix_idx)

        # unexpected type: return as-is
        return hess

    def get_dielectric_derivatives(self, atoms=None):
        """
        Call parent get_dielectric_derivatives() then zero per-atom entries for fixed atoms.
        The parent can return:
          - dipole_derivatives (array or list)  OR
          - (dipole_derivatives, polarizability_derivatives) tuple
        We handle both single-model and committee outputs.
        """
        res = super().get_dielectric_derivatives(atoms=atoms)

        if self._fix_indices is None:
            return res

        if atoms is None and self.atoms is None:
            return res
        if atoms is None:
            atoms = self.atoms

        fix_idx = self._resolve_fix_indices_for_atoms(atoms)
        if fix_idx is None or len(fix_idx) == 0:
            return res

        # helper to zero possible data shapes (list or ndarray)
        def _zero(data):
            return self._zero_per_atom_tensor(data, fix_idx)

        # res could be tuple (dipole_derivatives, polarizability_derivatives)
        if isinstance(res, tuple):
            dip, polar = res
            dip = _zero(dip)
            polar = _zero(polar)
            return dip, polar

        # otherwise single return value
        return _zero(res)
