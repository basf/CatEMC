#!/bin/bash
# Submit benchmarks for all worker counts.
# Results are collected in results.json after each job finishes.

WORKER_COUNTS="1 6 24 48"

for NW in $WORKER_COUNTS; do
    echo "Submitting benchmark with $NW workers..."
    qsub -v NUM_WORKERS=$NW submit_benchmark.sh
    sleep 2
done

echo ""
echo "All jobs submitted. Results will be saved to results.json."
echo "After all jobs finish, run: python plot_results.py"
