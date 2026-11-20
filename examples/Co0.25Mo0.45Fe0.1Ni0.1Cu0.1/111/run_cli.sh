#!/bin/bash
# CLI version of this example. Each command reads config.yaml, which holds
# everything that generate_initial_population.py,
# get_internal_partition_functions.py, gcemc_run.py and mbar_analysis.py
# define in Python. Run ../bulk/run_cli.sh first.
set -e

catemc check    config.yaml     # optional: show what the file resolves to
catemc generate config.yaml     # -> *_template.traj, *_input_pop.traj
catemc run      config.yaml     # -> *.db; run it again to continue
catemc analyze  config.yaml     # -> *_mbar_expectations.pkl, *_mbar_fes.pkl

# The plotting scripts in this folder read the analysis files:
# python plot_metals_spd_muN_vs_muH.py
# python plot_adsorbates_spd_muN_vs_muH.py
# python plot_surf_bond_matrix.py
