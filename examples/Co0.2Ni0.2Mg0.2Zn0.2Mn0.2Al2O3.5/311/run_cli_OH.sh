#!/bin/bash
# CLI version of the OH example. Each command reads config_OH.yaml, which
# holds everything that generate_initial_population_OH.py,
# get_internal_partition_functions.py, gcemc_run_OH.py and
# mbar_analysis_OH.py define in Python. Run ../bulk/run_cli.sh first.
set -e

catemc check    config_OH.yaml  # optional: show what the file resolves to
catemc generate config_OH.yaml  # -> *_template.traj, *_input_pop.traj
catemc run      config_OH.yaml  # -> *.db; run it again to continue
catemc analyze  config_OH.yaml  # -> *_mbar_expectations.pkl, *_mbar_fes.pkl
