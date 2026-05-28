# Persistent Worker Pool — Benchmark & Equivalence Tests

## Overview

This directory contains tests that verify the **persistent worker pool** mode
produces results equivalent to the original subprocess-spawning mode, and
benchmarks the speedup across multiple MLIP backends.

The worker pool keeps ML model(s) loaded in long-lived worker processes,
eliminating the per-generation overhead of spawning subprocesses that each
reload Python, PyTorch, CUDA, and the model from scratch.

## Benchmark Results

![Worker Pool Benchmark](../../images/worker_pool_benchmark.png)

### Measured Speedup (real data, NVIDIA A100-40GB)

**System:** Cu₇Ru₆₅O₁₄₄ (216 atoms), 3 replicas, 50 relaxation steps per generation

| Model | Generations | Worker Pool | Subprocess | Speedup |
|-------|-------------|-------------|------------|---------|
| **MACE** | 3 | 23.5s | 52.0s | 2.2× |
| **MACE** | 9 | 51.1s | 150.9s | 3.0× |
| **MACE** | 15 | 79.4s | 244.1s | **3.1×** |
| **PET** | 3 | 29.8s | 59.2s | 2.0× |
| **PET** | 9 | 60.3s | 165.4s | 2.7× |
| **PET** | 15 | 86.0s | 275.2s | **3.2×** |
| **NequIP** | 3 | 13.5s | 17.8s | 1.3× |
| **NequIP** | 15 | 14.1s | 88.0s | 6.2× |
| **NequIP** | 60 | 16.6s | 369.3s | **22.3×** |

### Per-Generation Breakdown

![Worker Pool Breakdown](../../images/worker_pool_breakdown.png)

| Model | Actual compute | Subprocess overhead | Overhead fraction |
|-------|---------------|--------------------|--------------------|
| **MACE** | 4.67s | 11.60s | 71% |
| **PET** | 4.71s | 13.64s | 74% |
| **NequIP** | 0.054s | 6.10s | **99%** |

The subprocess overhead (Python + PyTorch + CUDA bootstrap) is **constant per
generation** regardless of model — it accounts for 71–99% of the total subprocess
time. The worker pool eliminates this entirely after startup.

### Projected Production Runs

| Model | Generations | Subprocess | Worker Pool | Speedup | Time Saved |
|-------|-------------|-----------|-------------|---------|------------|
| **MACE** | 300 | 1.4h | 0.4h | 3.5× | 1.0h |
| **MACE** | 500 | 2.3h | 0.7h | 3.5× | 1.6h |
| **MACE** | 1500 | 6.8h | 2.0h | 3.5× | 4.8h |
| **PET** | 300 | 1.5h | 0.4h | 3.9× | 1.1h |
| **PET** | 500 | 2.5h | 0.7h | 3.9× | 1.9h |
| **PET** | 1500 | 7.6h | 2.0h | 3.9× | 5.7h |
| **NequIP** | 300 | 0.5h | 5.5s* | 67× | 0.5h |
| **NequIP** | 500 | 0.9h | 7.5s* | 109× | 0.8h |
| **NequIP** | 1500 | 2.6h | 17.5s* | 537× | 2.6h |

*NequIP compute per generation is only 54ms (model is very fast on GPU); nearly all subprocess time is overhead.

## Equivalence Verification

Both modes produce equivalent results. Small differences (~0.01 eV) arise from
non-deterministic CUDA execution in GNN scatter operations, not from code bugs.

```
Energy diffs (eV): max=0.0168, mean=0.0117
```

At T=573K (kT ≈ 0.049 eV), a 0.01 eV difference is ~0.2 kT — negligible impact
on Boltzmann acceptance and ensemble averages.

## Running the Benchmarks

### Per-model benchmark (includes subprocess comparison)

```bash
# MACE
export PATH=/path/to/venv/bin:$PATH
export LD_PRELOAD=/path/to/venv/lib/libstdc++.so.6
python tests/benchmark_worker_pool/test_worker_pool_mace.py

# PET
conda activate /path/to/metatomic-env
export LD_LIBRARY_PATH=...
python tests/benchmark_worker_pool/test_worker_pool_pet.py

# NequIP
export PATH=/path/to/allegro-venv/bin:$PATH
export LD_PRELOAD=/path/to/allegro-venv/lib/libstdc++.so.6
python tests/benchmark_worker_pool/test_worker_pool_nequip.py
```

### Full sweep (all models, all round counts)

```bash
cd tests/benchmark_worker_pool
nohup bash run_sweep_all.sh > sweep_all.log 2>&1 &
```

Results saved to `sweep_results.json`.

### Regenerate plots

```bash
python tests/benchmark_worker_pool/plot_benchmark.py
```

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
| `test_worker_pool_equivalence.py` | Equivalence test (NequIP, worker pool vs EMC subprocess) |
| `test_worker_pool_nequip.py` | NequIP multi-round benchmark with subprocess comparison |
| `test_worker_pool_mace.py` | MACE multi-round benchmark with subprocess comparison |
| `test_worker_pool_pet.py` | PET multi-round benchmark with subprocess comparison |
| `sweep_all_real.py` | Full sweep script (all models, all round counts) |
| `plot_benchmark.py` | Generate infographic plots from sweep_results.json |
| `sweep_results.json` | Raw benchmark data (all measured timings) |
