#!/bin/bash
# CLI version of quickstart.py. config.yaml is the file that
# "catemc init gibbs" writes, and template.traj its template structure, written with
#   ase build -x fcc -a 3.8 --cubic -r 2,2,2 Cu template.traj
set -e

catemc check    config.yaml     # optional: show what the file resolves to
catemc generate config.yaml     # -> *_input_pop.traj
catemc run      config.yaml     # -> *.db; run it again to continue
catemc analyze  config.yaml     # -> *_sro_data.pkl, *_mbar_dmu_estimates.pkl
