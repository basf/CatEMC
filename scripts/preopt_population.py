"""Pre-optimize the initial population with parallel GPU relaxation.

Handles X (vacancy marker) atoms: removes them before relaxation (NequIP
doesn't recognize X), then restores them at original positions afterward.

Usage:
    python preopt_population.py [--workers N]
"""
import sys
import os
import argparse
import multiprocessing as mp
import math

sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)

from ase import Atoms

MODEL = "/fs/cephfs/data/qm_inorganics/RuMOx_OER/NequIP-OAM-S-stage6-h100-plain.nequip.pt2"
FMAX = 0.05
DEFAULT_WORKERS = 6


def remove_markers(atoms):
    """Remove X marker atoms and return clean atoms + marker indices."""
    marker_indices = [i for i, atom in enumerate(atoms) if atom.symbol == "X"]
    non_marker_indices = [i for i, atom in enumerate(atoms) if atom.symbol != "X"]
    clean = atoms[non_marker_indices]
    return clean, marker_indices


def restore_markers(relaxed_atoms, marker_indices, original_atoms):
    """Reinsert X markers at their original indices to preserve indexing."""
    symbols = list(relaxed_atoms.symbols)
    positions = list(relaxed_atoms.positions)

    for idx in sorted(marker_indices):
        symbols.insert(idx, "X")
        positions.insert(idx, original_atoms[idx].position)

    restored = Atoms(symbols=symbols,
                     positions=positions,
                     cell=relaxed_atoms.cell,
                     pbc=relaxed_atoms.pbc)
    restored.info = original_atoms.info.copy()
    return restored


def _relax_batch(args):
    """Worker function: load model, relax assigned replicas."""
    worker_id, indices, input_fname, model_path, fmax = args

    import sys, os
    sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from ase.io import read
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter
    from nequip.integrations.ase import NequIPCalculator

    calc = NequIPCalculator.from_compiled_model(model_path, device="cuda")
    input_pop = read(input_fname, index=":")

    results = []
    for i in indices:
        original = input_pop[i].copy()
        clean, marker_indices = remove_markers(original)
        clean.calc = calc
        fcf = FrechetCellFilter(clean)
        opt = FIRE(fcf, logfile=None)
        opt.run(fmax=fmax, steps=500)
        clean.calc = None

        # Restore X markers at original positions
        restored = restore_markers(clean, marker_indices, original)
        results.append((i, restored, opt.nsteps, len(marker_indices)))
        print(f"  [Worker {worker_id}] Replica {i}: relaxed ({opt.nsteps} steps, {len(marker_indices)} vacancies)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Number of parallel workers (default: %(default)s)")
    args = parser.parse_args()

    cwd = os.path.dirname(os.path.abspath(__file__))
    input_fname = os.path.join(cwd, "Cu0.1Ru0.9Ox_bulk_input_pop_unrelaxed.traj")
    output_fname = os.path.join(cwd, "Cu0.1Ru0.9Ox_bulk_input_pop.traj")

    from ase.io import read, write
    input_pop = read(input_fname, index=":")
    n = len(input_pop)
    num_workers = min(args.workers, n)

    print(f"Pre-optimizing {n} replicas with {num_workers} parallel workers on GPU")
    print(f"  Model: /fs/cephfs/data/qm_inorganics/RuMOx_OER/NequIP-OAM-S-stage6-h100-plain.nequip.pt2")
    print(f"  FMAX: 0.05")
    print(f"  X (vacancy) atoms will be stripped before relaxation and restored after")

    # Split indices into chunks
    chunk_size = math.ceil(n / num_workers)
    tasks = []
    for w in range(num_workers):
        start = w * chunk_size
        end = min(start + chunk_size, n)
        if start >= n:
            break
        indices = list(range(start, end))
        tasks.append((w, indices, input_fname, MODEL, FMAX))

    # Run in parallel with spawn context (CUDA-safe)
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers) as pool:
        all_results = pool.map(_relax_batch, tasks)

    # Reassemble in order
    relaxed = [None] * n
    for batch in all_results:
        for idx, atoms, nsteps, n_vac in batch:
            relaxed[idx] = atoms

    write(output_fname, relaxed)
    print()
    print(f"Written {n} pre-optimized replicas to {output_fname}")


if __name__ == "__main__":
    main()
