"""Structured logging for CatEMC EMC simulations.

Provides:
- Human-readable stdout at 4 verbosity levels (SILENT/SUMMARY/DETAILED/TRACE)
- Machine-parseable JSONL metrics file for live observer dashboards
"""
import json
import time
import os
import numpy as np
from datetime import datetime
from ase.parallel import parprint


# Verbosity levels
SILENT = 0
SUMMARY = 1
DETAILED = 2
TRACE = 3

_LEVEL_MAP = {
    'SILENT': SILENT, 'SUMMARY': SUMMARY,
    'DETAILED': DETAILED, 'TRACE': TRACE,
    0: SILENT, 1: SUMMARY, 2: DETAILED, 3: TRACE,
    True: TRACE, False: SILENT,
}


class EMCLogger:
    """Structured logger for EMC runs.

    Replaces ad-hoc parprint calls with level-gated output and
    optional JSONL metrics file for live monitoring.

    Parameters
    ----------
    name : str
        System name (used for JSONL filename).
    calculation_dir : str
        Directory for the JSONL metrics file (always on persistent storage).
    verbosity : str or int
        One of 'SILENT', 'SUMMARY', 'DETAILED', 'TRACE' or 0-3.
        Also accepts True (TRACE) / False (SILENT) for backward compat.
    metrics_file : bool
        If True, write per-generation JSONL to {calculation_dir}/{name}_metrics.jsonl
    """

    def __init__(self, name, calculation_dir, verbosity='SUMMARY', metrics_file=True):
        self.name = name
        self.calculation_dir = calculation_dir
        self.verbosity = _LEVEL_MAP.get(verbosity, SUMMARY)
        self.metrics_file = metrics_file

        # Per-generation counters (reset each gen)
        self._accepted = {}  # {'intra': N, 'inter': N, 'exchange': N}
        self._total = {}     # same keys
        self._gen_start_time = None

        # JSONL file handle (opened lazily)
        self._jsonl_path = os.path.join(calculation_dir, f'{name}_metrics.jsonl')
        self._jsonl_fh = None

    def _ensure_jsonl(self):
        """Lazily open the JSONL file for appending."""
        if self._jsonl_fh is None and self.metrics_file:
            os.makedirs(self.calculation_dir, exist_ok=True)
            self._jsonl_fh = open(self._jsonl_path, 'a', buffering=1)  # line-buffered

    def close(self):
        """Close the JSONL file handle."""
        if self._jsonl_fh is not None:
            self._jsonl_fh.close()
            self._jsonl_fh = None

    # ------------------------------------------------------------------
    # Generation lifecycle
    # ------------------------------------------------------------------

    def begin_generation(self):
        """Mark the start of a new generation (for timing)."""
        self._gen_start_time = time.perf_counter()
        self._accepted = {}
        self._total = {}

    def log_generation(self, gen_num, elapsed_s, pop=None,
                       temperature_key='temperature', target_metal_species=None):
        """Log end-of-generation summary.

        Parameters
        ----------
        gen_num : int
            Current generation number.
        elapsed_s : float
            Wall time for this generation in seconds.
        pop : list of Atoms, optional
            Current population for energy/composition metrics.
        temperature_key : str
            Key to extract temperature from atoms.info['key_value_pairs'].
        target_metal_species : list of str, optional
            Metal species to track for composition metrics.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Compute metrics from population (read-only)
        energy_stats = {}
        composition = {}
        temp_energy = []

        if pop is not None:
            from ase.ga import get_raw_score
            energies = [-1.0*get_raw_score(a) for a in pop]
            energies_arr = np.array(energies)
            energy_stats = {
                'mean': float(np.mean(energies_arr)),
                'min': float(np.min(energies_arr)),
                'max': float(np.max(energies_arr)),
                'std': float(np.std(energies_arr)),
            }

            # Temperature vs energy pairs
            for a, e in zip(pop, energies):
                T = a.info.get('key_value_pairs', {}).get(temperature_key, None)
                if T is not None:
                    temp_energy.append([float(T), float(e)])

            # Composition: fraction of target metals
            if target_metal_species:
                fracs = {s: [] for s in target_metal_species}
                for a in pop:
                    symbols = [atom.symbol for atom in a if atom.symbol != 'X']
                    n_total = len(symbols)
                    if n_total > 0:
                        for s in target_metal_species:
                            fracs[s].append(symbols.count(s) / n_total)
                composition = {
                    s: {'mean': float(np.mean(fracs[s])),
                        'std': float(np.std(fracs[s]))}
                    for s in target_metal_species if fracs[s]
                }

        # Total acceptance rate
        total_acc = sum(self._accepted.values())
        total_moves = sum(self._total.values())
        acc_rate = total_acc / total_moves if total_moves > 0 else 0.0

        # --- STDOUT ---
        if self.verbosity >= SUMMARY:
            e_str = f"Ē={energy_stats['mean']:.1f}" if energy_stats else ""
            line = (f"[{now_str}] Gen {gen_num} │ {elapsed_s:.1f}s │ "
                    f"acc {acc_rate:.0%} ({total_acc}/{total_moves}) │ {e_str}")
            parprint(line)

        if self.verbosity >= DETAILED and self._total:
            parts = []
            for move_type in sorted(self._total.keys()):
                a = self._accepted.get(move_type, 0)
                t = self._total[move_type]
                parts.append(f"  {move_type}: {a}/{t}")
            parprint("  " + " | ".join(parts))

        # --- JSONL ---
        if self.metrics_file:
            self._ensure_jsonl()
            record = {
                'gen': gen_num,
                't': time.time(),
                'elapsed_s': round(elapsed_s, 3),
                'n_accepted': dict(self._accepted),
                'n_total': dict(self._total),
                'acc_rate': round(acc_rate, 4),
            }
            if energy_stats:
                record['energy'] = {k: round(v, 4) for k, v in energy_stats.items()}
            if composition:
                record['composition'] = composition
            if temp_energy:
                record['temp_energy'] = [[round(t, 1), round(e, 4)] for t, e in temp_energy]
            self._jsonl_fh.write(json.dumps(record) + '\n')

    # ------------------------------------------------------------------
    # Per-replica move logging (TRACE level)
    # ------------------------------------------------------------------

    def log_move(self, move_type, accepted, message=None):
        """Record a move result (intra/inter/exchange/mu_walk).

        Always updates counters. Prints message only at TRACE level.

        Parameters
        ----------
        move_type : str
            One of 'intra', 'inter', 'exchange', 'mu_walk'.
        accepted : bool
            Whether the move was accepted.
        message : str, optional
            Detailed message to print at TRACE verbosity.
        """
        self._total[move_type] = self._total.get(move_type, 0) + 1
        if accepted:
            self._accepted[move_type] = self._accepted.get(move_type, 0) + 1

        if message and self.verbosity >= TRACE:
            parprint(message)

    # ------------------------------------------------------------------
    # Event logging (non-per-gen events)
    # ------------------------------------------------------------------

    def log_event(self, category, message, level=SUMMARY):
        """Log a non-generation event (scratch-db, trim, restart, etc.).

        Parameters
        ----------
        category : str
            Event category (for JSONL tagging).
        message : str
            Human-readable message.
        level : int
            Minimum verbosity level to print this event.
        """
        if self.verbosity >= level:
            parprint(message)

        if self.metrics_file:
            self._ensure_jsonl()
            record = {
                'event': category,
                't': time.time(),
                'message': message,
            }
            self._jsonl_fh.write(json.dumps(record) + '\n')
