# CatEMC I/O Reduction Benchmark

## Problem

CephFS metadata operations (open/close/fsync) are slow (~1-2ms per op).
The original EMC code performed **96+ file operations per generation** when using
`worker_pool`:
- 48 offspring writes (`atoms_{rep}.traj`)
- 48 offspring reads (read back immediately after writing)
- 48 trajectory appends (`emc_{rep}.traj`) every generation

With 48 replicas this added ~80-85s of I/O overhead per generation on CephFS,
while actual GPU compute (NequIP eval) took only ~5-20s.

## Solution (branch: `perf/reduce-cephfs-io`)

Two optimizations in `catemc/emc.py`:

1. **In-memory offspring passing** (`_offspring_cache` dict):
   - When `worker_pool` is active, offspring atoms are stored in a Python dict
   - Eliminates 96 file round-trips per generation
   - Falls back to file-based passing when using subprocess mode (no worker_pool)

2. **`traj_write_interval` parameter** (default=1, backwards compatible):
   - `emc_{rep}.traj` files appended only every N generations
   - Reduces 48 file writes per gen to 48 writes every N gens
   - Simulation state still maintained in memory + SQLite DB

3. **Staggered worker startup** (in `worker_pool.py`):
   - 0.5s delay between spawning workers when >4 workers
   - Prevents all workers from loading PyTorch + model simultaneously from CephFS

## Test Files

| File | Purpose |
|------|---------|
| `validate_patch.py` | 6-replica correctness test (patched vs original, fixed seed) |
| `validate_48rep.py` | 48-replica 3-way test (old vs new-1w vs new-48w) |
| `sweep_workers.py` | Worker count scaling sweep (1,6,12,18,24,30,36,42,48) |
| `submit_validate.sh` | PBS submit for 6-replica test |
| `submit_validate_48.sh` | PBS submit for 48-replica test |
| `submit_sweep.sh` | PBS submit for worker scaling sweep |

## Results

### Correctness Validation (6 replicas, 5 gens, seed=42)
- All 40 acceptance decisions: **IDENTICAL**
- Max energy difference: **3.1e-4 eV** (GPU floating-point non-determinism)

### 3-Way Timing (48 replicas, 5 gens)
| Configuration | Time/gen | Speedup |
|---------------|----------|---------|
| OLD code, 1 worker | 20.7 s/gen | 1.0x |
| NEW code, 1 worker, traj every 5 | 20.0 s/gen | 1.0x |
| NEW code, 48 workers, traj every 5 | 4.9 s/gen | **4.3x** |

### Worker Scaling Sweep (48 replicas, 30 gens, traj every 30)
*Results in `sweep_results.json` after running `submit_sweep.sh`*

## How to Reproduce

```bash
cd /path/to/CatEMC/tests/benchmark_io_reduction

# Correctness validation (6 replicas, ~5 min)
qsub submit_validate.sh

# 3-way timing comparison (48 replicas, ~15 min)
qsub submit_validate_48.sh

# Full worker scaling sweep (48 replicas × 9 configs × 30 gens, ~2-4h)
qsub submit_sweep.sh
```

## Key Findings

1. **In-memory offspring** eliminates unnecessary CephFS round-trips but shows
   minimal timing improvement when CephFS is not under load (~1.0x at low load).
   Under heavy CephFS load (multiple concurrent jobs), the improvement is much
   larger (up to 8-9x observed).

2. **Parallelism (48 workers)** provides ~4-5x speedup on compute-bound
   generations (swap + single-point eval). MD generations (every 10th) benefit
   more from parallelism.

3. **GPU non-determinism** causes ~0.3 meV energy variations between separate
   process invocations. This flips ~5% of borderline accept/reject decisions
   but does not affect ensemble statistics.

## Environment

- GPU: NVIDIA H100 NVL (95830 MiB)
- Model: NequIP-OAM-S-stage6 (~3.7ms/eval for 205 atoms)
- Filesystem: CephFS (shared, metadata-limited)
- System: Cu0.1Ru0.9Ox bulk, 48 replicas, 216 atoms (205 real + 11 vacancies)
- Python: 3.12, NequIP 0.18, PyTorch 2.12.0+cu126
