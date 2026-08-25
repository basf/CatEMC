import warnings

import numpy as np
from ase.filters import UnitCellFilter
from ase.optimize.optimize import Optimizer


class _SQNMCore:

    def __init__(self, ndim, nhist_max=10, initial_step_size=-0.01,
                 alpha_min=1e-3, eps_subsp=1e-3):
        self.ndim = int(ndim)
        self.nhist_max = min(int(nhist_max), self.ndim)
        self.alpha_min = float(alpha_min)
        self.eps_subsp = float(eps_subsp)
        self.estimate_step_size = initial_step_size <= 0
        self.alpha = abs(float(initial_step_size))
        self.x_history = []
        self.g_history = []
        self.direction = np.zeros(self.ndim)
        self.expected_x = None
        self.prev_energy = None
        self.prev_gradient = None
        self.gainratio = 0.0
        self.h_eval = np.empty(0)

    def step(self, x, energy, gradient):
        x = np.asarray(x, dtype=float).reshape(-1)
        gradient = np.asarray(gradient, dtype=float).reshape(-1)
        if x.size != self.ndim or gradient.size != self.ndim:
            raise ValueError('SQNM coordinate and gradient dimensions do not match')
        if np.linalg.norm(gradient) < 1e-12:
            self.direction = np.zeros_like(gradient)
            self.expected_x = x.copy()
            self.prev_energy = float(energy)
            self.prev_gradient = gradient.copy()
            return self.direction.copy()

        self.x_history.append(x.copy())
        self.g_history.append(gradient.copy())
        keep = self.nhist_max + 1
        if len(self.x_history) > keep:
            self.x_history = self.x_history[-keep:]
            self.g_history = self.g_history[-keep:]

        nhist = min(len(self.x_history) - 1, self.nhist_max)
        if nhist == 0:
            direction = -self.alpha * gradient
        else:
            if self.expected_x is not None and np.max(np.abs(x - self.expected_x)) > 1e-8:
                warnings.warn('SQNM received coordinates different from the previous proposed step.',
                              RuntimeWarning, stacklevel=2)

            self._update_step_size(float(energy), gradient)
            dx = np.diff(np.asarray(self.x_history), axis=0)[-nhist:].T
            dg = np.diff(np.asarray(self.g_history), axis=0)[-nhist:].T
            norms = np.linalg.norm(dx, axis=0)
            valid = norms > 1e-14
            dx = dx[:, valid]
            dg = dg[:, valid]
            norms = norms[valid]

            if dx.shape[1] == 0:
                direction = -self.alpha * gradient
            else:
                normalized_dx = dx / norms
                overlap = normalized_dx.T @ normalized_dx
                s_eval, s_evec = np.linalg.eigh(overlap)
                if s_eval[-1] <= 0:
                    direction = -self.alpha * gradient
                else:
                    keep_mask = s_eval / s_eval[-1] > self.eps_subsp
                    s_eval = s_eval[keep_mask]
                    s_evec = s_evec[:, keep_mask]
                    dr_subsp = normalized_dx @ s_evec / np.sqrt(s_eval)
                    df_subsp = (dg / norms) @ s_evec / np.sqrt(s_eval)
                    h_subsp = 0.5 * (df_subsp.T @ dr_subsp + dr_subsp.T @ df_subsp)
                    h_eval, h_evec_subsp = np.linalg.eigh(h_subsp)
                    h_evec = dr_subsp @ h_evec_subsp
                    residual = np.linalg.norm(
                        -h_evec * h_eval + df_subsp @ h_evec_subsp, axis=0)
                    h_eval = np.sqrt(h_eval**2 + residual**2)
                    self.h_eval = h_eval.copy()

                    projection = h_evec.T @ gradient
                    outside = gradient - h_evec @ projection
                    direction = -(self.alpha * outside +
                                  h_evec @ (projection / h_eval))

        self.direction = direction
        self.expected_x = x + direction
        self.prev_energy = float(energy)
        self.prev_gradient = gradient.copy()
        return direction.copy()

    def _update_step_size(self, energy, gradient):
        if self.prev_gradient is None:
            return
        prev_norm = np.linalg.norm(self.prev_gradient)
        if prev_norm < 1e-14:
            return

        if self.estimate_step_size:
            denom = self.alpha * prev_norm
            l1 = (energy - self.prev_energy + self.alpha * prev_norm**2) / \
                 (0.5 * self.alpha**2 * prev_norm**2)
            l2 = np.linalg.norm(gradient - self.prev_gradient) / denom
            curvature = max(l1, l2)
            if np.isfinite(curvature) and curvature > 0:
                self.alpha = max(self.alpha_min, 1.0 / curvature)
            self.estimate_step_size = False
            return

        denom = 0.5 * np.dot(self.direction, self.prev_gradient)
        if abs(denom) < 1e-20:
            return
        self.gainratio = (energy - self.prev_energy) / denom
        if self.gainratio < 0.5:
            self.alpha = max(self.alpha_min, 0.65 * self.alpha)
        elif self.gainratio > 1.05:
            self.alpha *= 1.05

    def rescale_last_step(self, x, factor):
        self.direction *= factor
        self.expected_x = np.asarray(x, dtype=float).reshape(-1) + self.direction

    def lower_bound(self):
        if self.prev_energy is None or self.prev_gradient is None or self.h_eval.size == 0:
            return 0.0
        curvature = np.min(self.h_eval)
        if curvature <= 0:
            return None
        return self.prev_energy - 0.5 * np.dot(self.prev_gradient, self.prev_gradient) / curvature

    def get_state(self):
        return dict(alpha=self.alpha,
                    estimate_step_size=self.estimate_step_size,
                    x_history=self.x_history,
                    g_history=self.g_history,
                    direction=self.direction,
                    expected_x=self.expected_x,
                    prev_energy=self.prev_energy,
                    prev_gradient=self.prev_gradient,
                    gainratio=self.gainratio,
                    h_eval=self.h_eval)

    def set_state(self, state):
        self.alpha = float(state['alpha'])
        self.estimate_step_size = bool(state['estimate_step_size'])
        self.x_history = [np.asarray(x, dtype=float) for x in state['x_history']]
        self.g_history = [np.asarray(g, dtype=float) for g in state['g_history']]
        self.direction = np.asarray(state['direction'], dtype=float)
        expected_x = state['expected_x']
        self.expected_x = None if expected_x is None else np.asarray(expected_x, dtype=float)
        self.prev_energy = state['prev_energy']
        prev_gradient = state['prev_gradient']
        self.prev_gradient = None if prev_gradient is None else np.asarray(prev_gradient, dtype=float)
        self.gainratio = float(state['gainratio'])
        self.h_eval = np.asarray(state['h_eval'], dtype=float)


class SQNM(Optimizer):
    """Stabilized quasi-Newton optimizer with variable-cell support.

    The optimizer follows the SQNM subspace construction and stabilization of
    Schaefer et al. and the variable-cell coordinate transformation of Gubler
    et al. A UnitCellFilter or FrechetCellFilter activates variable-cell mode.

    Parameters
    ----------
    atoms : ase.Atoms or ase.filters.UnitCellFilter
        Object to optimize.
    restart : str or None, default None
        ASE JSON restart file.
    logfile : file object or str or None, default '-'
        Optimizer log file.
    trajectory : str or None, default None
        ASE trajectory file.
    maxstep : float or None, default 0.2
        Maximum Cartesian displacement of any atom in one step, in angstrom.
    nhist_max : int, default 10
        Maximum number of displacement/gradient differences retained.
    initial_step_size : float, default -0.01
        Initial SQNM step size. A non-positive value enables the automatic
        initial-step estimate and its absolute value is used for the first step.
    alpha_min : float, default 1e-3
        Lower bound for the adaptive SQNM step size.
    eps_subsp : float, default 1e-3
        Relative overlap-eigenvalue cutoff defining the significant subspace.
    lattice_weight : float, default 2.0
        Scaling of lattice degrees of freedom in variable-cell mode.
    master : bool or None, default None
        ASE trajectory-writing rank selection.
    append_trajectory : bool, default False
        Append to an existing trajectory instead of overwriting it.
    **kwargs
        Additional arguments passed to :class:`ase.optimize.Optimizer`.
    """

    defaults = dict(Optimizer.defaults, maxstep=0.2)

    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=None, nhist_max=10, initial_step_size=-0.01,
                 alpha_min=1e-3, eps_subsp=1e-3, lattice_weight=2.0,
                 master=None, append_trajectory=False, **kwargs):
        self.maxstep = self.defaults['maxstep'] if maxstep is None else float(maxstep)
        self.nhist_max = int(nhist_max)
        self.initial_step_size = float(initial_step_size)
        self.alpha_min = float(alpha_min)
        self.eps_subsp = float(eps_subsp)
        self.lattice_weight = float(lattice_weight)
        if self.maxstep <= 0:
            raise ValueError('maxstep must be positive')
        if self.nhist_max < 1:
            raise ValueError('nhist_max must be positive')
        if self.initial_step_size == 0:
            raise ValueError('initial_step_size cannot be zero')
        if self.alpha_min <= 0 or self.eps_subsp <= 0 or self.lattice_weight <= 0:
            raise ValueError('alpha_min, eps_subsp and lattice_weight must be positive')
        self._variable_cell = isinstance(atoms, UnitCellFilter)
        self._physical_atoms = atoms.atoms if self._variable_cell else atoms
        self._initial_cell = None
        self._lattice_transformer = None
        self._lattice_transformer_inv = None
        self._fluctuation = 0.0

        if master is not None:
            kwargs['master'] = master
        kwargs['append_trajectory'] = append_trajectory
        super().__init__(atoms, restart=restart, logfile=logfile,
                         trajectory=trajectory, **kwargs)

    def initialize(self):
        ndim = 3 * len(self.atoms)
        self._core = _SQNMCore(ndim, self.nhist_max, self.initial_step_size,
                               self.alpha_min, self.eps_subsp)
        if self._variable_cell:
            self._initialize_variable_cell()

    def _initialize_variable_cell(self):
        nat = len(self._physical_atoms)
        self._initial_cell = np.asarray(self._physical_atoms.get_cell(), dtype=float).T.copy()
        norms = np.linalg.norm(self._initial_cell, axis=0)
        if np.any(norms <= 0):
            raise ValueError('Variable-cell SQNM requires three non-zero lattice vectors')
        self._lattice_transformer = (np.diag(1.0 / norms) *
                                     self.lattice_weight * np.sqrt(nat))
        self._lattice_transformer_inv = np.linalg.inv(self._lattice_transformer)

    def step(self, forces=None):
        if self._variable_cell:
            self._step_variable_cell()
        else:
            self._step_fixed_cell(forces)
        self.dump(self._restart_state())

    def _step_fixed_cell(self, forces=None):
        gradient = self._get_gradient(forces)
        x = self.optimizable.get_x()
        energy = self.optimizable.get_value()
        step = self._core.step(x, energy, gradient)
        step = self._limit_atomic_step(step)
        self.optimizable.set_x(x + step)
        self._update_force_noise((-gradient.reshape(-1, 3)).T)

    def _step_variable_cell(self):
        atoms = self._physical_atoms
        nat = len(atoms)
        positions = atoms.get_positions().T
        forces = atoms.get_forces().T
        energy = self.optimizable.get_value()
        cell = np.asarray(atoms.get_cell(), dtype=float).T
        cell_inv = np.linalg.inv(cell)
        initial_cell_inv = np.linalg.inv(self._initial_cell)

        pressure = float(getattr(self.atoms, 'scalar_pressure', 0.0))
        stress = atoms.get_stress(voigt=False)
        lattice_force = (-atoms.get_volume() *
                         (stress + pressure * np.eye(3)) @ cell_inv.T)

        q = (self._initial_cell @ cell_inv @ positions).reshape(3 * nat)
        grad_q = (-(cell @ initial_cell_inv) @ forces).reshape(3 * nat)
        cell_scaled = (cell @ self._lattice_transformer).reshape(9)
        grad_cell = (-lattice_force @ self._lattice_transformer_inv).reshape(9)
        x = np.concatenate((q, cell_scaled))
        gradient = np.concatenate((grad_q, grad_cell))
        step = self._core.step(x, energy, gradient)

        trial = x + step
        trial_cell = trial[3 * nat:].reshape(3, 3) @ self._lattice_transformer_inv
        trial_positions = (trial_cell @ initial_cell_inv @
                           trial[:3 * nat].reshape(3, nat))
        displacements = (trial_positions - positions).T
        max_displacement = np.linalg.norm(displacements, axis=1).max(initial=0.0)
        if max_displacement > self.maxstep:
            factor = self.maxstep / max_displacement
            self._core.rescale_last_step(x, factor)
            trial = x + self._core.direction
            trial_cell = trial[3 * nat:].reshape(3, 3) @ self._lattice_transformer_inv
            trial_positions = (trial_cell @ initial_cell_inv @
                               trial[:3 * nat].reshape(3, nat))

        if not np.all(np.isfinite(trial_cell)) or np.linalg.det(trial_cell) <= 0:
            raise RuntimeError('SQNM produced an invalid simulation cell')
        atoms.set_cell(trial_cell.T, scale_atoms=False)
        atoms.set_positions(trial_positions.T)
        self._update_force_noise(forces)

    def _limit_atomic_step(self, step):
        step_matrix = np.asarray(step).reshape(-1, 3)
        max_displacement = np.linalg.norm(step_matrix, axis=1).max(initial=0.0)
        if max_displacement > self.maxstep:
            factor = self.maxstep / max_displacement
            x = self.optimizable.get_x()
            self._core.rescale_last_step(x, factor)
            return self._core.direction.copy()
        return step

    def _update_force_noise(self, forces):
        if forces.size == 0:
            return
        noise = np.linalg.norm(np.sum(forces, axis=1)) / np.sqrt(forces.size)
        self._fluctuation = noise if self._fluctuation == 0 else \
            0.8 * self._fluctuation + 0.2 * noise
        max_force = np.max(np.abs(forces))
        if max_force > 0 and self._fluctuation > 0.2 * max_force:
            warnings.warn('Force noise is large relative to the maximum force; '
                          'SQNM convergence may be affected.', RuntimeWarning,
                          stacklevel=2)

    def lower_bound(self):
        """Return the SQNM local-minimum lower-bound estimate when available."""
        return self._core.lower_bound()

    def _restart_state(self):
        return dict(core=self._core.get_state(),
                    initial_cell=self._initial_cell,
                    fluctuation=self._fluctuation)

    def read(self):
        state = self.load()
        ndim = 3 * len(self.atoms)
        self._core = _SQNMCore(ndim, self.nhist_max, self.initial_step_size,
                               self.alpha_min, self.eps_subsp)
        self._core.set_state(state['core'])
        self._fluctuation = float(state.get('fluctuation', 0.0))
        if self._variable_cell:
            self._initial_cell = np.asarray(state['initial_cell'], dtype=float)
            nat = len(self._physical_atoms)
            norms = np.linalg.norm(self._initial_cell, axis=0)
            self._lattice_transformer = (np.diag(1.0 / norms) *
                                         self.lattice_weight * np.sqrt(nat))
            self._lattice_transformer_inv = np.linalg.inv(self._lattice_transformer)
