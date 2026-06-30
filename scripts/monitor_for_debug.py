#!/usr/bin/env python3

from ase.db import connect

import sys

if len(sys.argv)!=2:
    print('use python monitor_for_debug.py <filename>')
    sys.exit(1)


filename = sys.argv[-1]

db = connect(filename)

for row in db.select(relaxed=1):
    print(f"gen={row.emc_step}, t={row.key_value_pairs['temperature']}, replica {row.key_value_pairs['replica']}, valence={row.key_value_pairs['dopant_valence']}, en={row.potential_energy:.2f} eV, accepted={row.key_value_pairs['accepted']}")
