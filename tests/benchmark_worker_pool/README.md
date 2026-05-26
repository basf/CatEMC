# Persistent Worker Pool — Benchmark & Equivalence Tests

## Overview

This directory contains tests that verify the **persistent worker pool** mode
produces results equivalent to the original subprocess-spawning mode, and
benchmarks the speedup.

The worker pool keeps ML model(s) loaded in long-lived worker processes,
eliminating the per-generation overhead of spawning subprocesses that each
reload the model from disk.

## Benchmark Results (2026-05-27)

**System:** 48 replicas, Cu₇Ru₆₅O₁₄₄ (216 atoms), NequIP-OAM-S compiled model  
**Hardware:** NVIDIA A100-SXM4-40GB  
**Environment:** Python 3.11, `/mnt/nfs/venvs/allegro`

| Metric | Worker Pool (20 workers) | Subprocess (20 procs/device) | Speedup |
|--------|--------------------------|------------------------------|---------|
| Startup | 11.6s (one-time) | 0s | — |
| Gen 1 (incl. CUDA JIT warmup) | 8.09s | 45.95s | 5.7x |
| Steady-state (gen 2–10) | **1.16–1.24s** | **~41.2s** | **35x** |
| Mean gen time (10 gen) | 1.88s | 42.81s | 23x |
| Throughput | 25.5 struct/s | 1.1 struct/s | 23x |

**Projected 5000-generation production run:**
- Worker pool: 11.6s startup + 8s warmup + 4999 × 1.2s ≈ **100 min**
- Subprocess: 5000 × 41s ≈ **57 hours**
- **Improvement: 57h → 1.7h (34× faster)**

## Equivalence Verification

Both modes were run on identical input structures (5 replicas, `run_md=False`
for deterministic relaxation). Per-replica energy comparison:

```
Worker pool energies: [-2744.5026, -2744.5361, -2742.7101, -2745.9040, -2739.9653]
Subprocess energies:  [-2744.4941, -2744.5529, -2742.7197, -2745.9172, -2739.9759]

Energy diffs (eV):  max=0.016799, mean=0.011748
```

### Why are there small differences?

The ~0.01 eV differences are **not** a bug. They arise from:

1. **Non-deterministic CUDA execution** — GPU scatter/atomics operations in
   graph neural networks (NequIP/Allegro) produce slightly different results
   across runs due to floating-point non-associativity.
2. **LBFGS sensitivity** — the optimizer converges to slightly different local
   configurations since forces near the convergence threshold (fmax=0.10 eV/Å)
   have residual floating-point variation.

We verified this by running the **same code path twice** on the **same input**:
```
Energy after relax: orig=-2744.473255, round-trip=-2744.482402, diff=0.009146 eV
```
The traj file round-trip preserves positions exactly (0.00 Å difference), yet
relaxation still produces ~0.009 eV variation — confirming this is inherent GPU
non-determinism.

**Impact on MC sampling:** At T=573K, kT ≈ 0.049 eV. A 0.01 eV difference is
~0.2 kT, which has negligible impact on Boltzmann acceptance probabilities
and ensemble averages.

## Running the Tests

### Prerequisites

- NVIDIA GPU with CUDA support
- NequIP compiled model (`.nequip.pt2`)
- CatEMC on `PYTHONPATH`
- Initial population trajectory file
- The `calculate_fitness.py` script in `CALC_SCRIPT_DIR` (for subprocess mode)

### Equivalence Test

```bash
cd /path/to/CatEMC
export PYTHONPATH=$PWD:$PYTHONPATH
export PATH=/path/to/venv/bin:$PATH
export LD_PRELOAD=/path/to/venv/lib/libstdc++.so.6

# Set paths (or use defaults for the BASF oracle-a100 node)
export COMPILED_MODEL=/path/to/model.nequip.pt2
export POPULATION_FILE=/path/to/input_pop.traj
export CALC_SCRIPT_DIR=/path/to/dir/with/calculate_fitness.py

python tests/benchmark_worker_pool/test_worker_pool_equivalence.py
```

Expected output:
```
  PASS: max energy difference (0.0168 eV) < tolerance (0.05 eV)
  Note: ~0.01 eV differences are expected from non-deterministic CUDA atomics
```

### Performance Benchmark

```bash
python benchmark_workers.py --mode both --generations 10 --replicas 48
```

This runs both modes and reports the speedup factor.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Main Process (EMC/GEEMC/GCEMC)                         │
│                                                         │
│  pool = WorkerPool(factory_module, factory_func, ...)   │
│  results = pool.evaluate(atoms_list, run_md=False)      │
└───────────────┬─────────────────────────────────────────┘
                │ task_queue (multiprocessing.Queue)
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐
│Worker 0│ │Worker 1│ │  ...   │  N workers (spawn context)
│ Model  │ │ Model  │ │ Model  │  Each loads model ONCE at startup
│ loaded │ │ loaded │ │ loaded │  Stays alive across generations
└───┬────┘ └───┬────┘ └───┬────┘
    │           │           │
    └───────────┼───────────┘
                │ result_queue
                ▼
        Results in submission order
```

## Files

| File | Description |
|------|-------------|
| `test_worker_pool_equivalence.py` | Equivalence test: same results from both modes |
| `../../benchmark_workers.py` | Performance benchmark comparing both modes |
| `../../catemc/worker_pool.py` | WorkerPool implementation |
| `../../catemc/calculation_templates.py` | Contains `calculate_bulk_fitness_inprocess()` |
