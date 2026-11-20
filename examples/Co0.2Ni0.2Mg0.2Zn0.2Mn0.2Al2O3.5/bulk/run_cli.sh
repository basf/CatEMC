#!/bin/bash
# CLI version of this example. Each command reads config.yaml, which holds
# everything that generate_initial_population.py, geemc_run.py and
# mbar_analysis.py define in Python.
set -e

# Gibbs-ensemble EMC (composition fluctuates around the nominal one)
catemc check    config.yaml     # optional: show what the file resolves to
catemc generate config.yaml     # -> *_template.traj, *_input_pop.traj
catemc run      config.yaml     # -> *.db; run it again to continue
catemc analyze  config.yaml     # -> *_sro_data.pkl, *_mbar_dmu_estimates.pkl

# Canonical-ensemble EMC (fixed composition); uncomment to run
# catemc generate config_canonical.yaml
# catemc run      config_canonical.yaml
# catemc analyze  config_canonical.yaml
