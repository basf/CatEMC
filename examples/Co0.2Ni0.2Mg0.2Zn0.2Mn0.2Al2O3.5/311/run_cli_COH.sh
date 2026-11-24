#!/bin/bash
# CLI version of the COH example. Each command reads config_COH.yaml, which
# holds everything that generate_initial_population_COH.py,
# get_internal_partition_functions.py, gcemc_run_COH.py and
# mbar_analysis_COH.py define in Python. Run ../bulk/run_cli.sh first and
# copy the metal chemical potentials it prints into chemical_potentials.metals.
set -e

catemc check    config_COH.yaml  # optional: show what the file resolves to
catemc generate config_COH.yaml  # -> *_input_pop.traj
catemc run      config_COH.yaml  # -> *.db; run it again to continue
catemc analyze  config_COH.yaml  # -> *_mbar_expectations.pkl, *_mbar_fes.pkl
