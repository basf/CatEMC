"""Generate initial population for Cu0.1Ru0.9Ox bulk EMC."""
from catemc.generate import generate_population
from ase.io import read, write
import numpy as np
import os

# --- Configuration ---
dopant = "Cu"
host_metal = "Ru"
dopant_fraction = 0.1
structure = "bulk"
template_cif = "/fs/cephfs/data/qm_inorganics/RuMOx_OER/RuO2.cif"
supercell = [3, 3, 4]

target_metal_species = [dopant, host_metal]
target_metal_composition = {dopant: dopant_fraction, host_metal: 0.9}
target_nonmetal_species = ["O", "X"]
dopant_valences = [1, 2, 3, 4]
replicas_per_valence = 6
temperatures = [300.0, 381.68, 485.59, 617.8, 786.0, 1000.0]

cwd = os.path.dirname(os.path.abspath(__file__))
comp_name = f"{dopant}{dopant_fraction}{host_metal}0.9Ox"
system_name = f"{comp_name}_{structure}"

# --- Build supercell ---
template = read(template_cif)
template *= tuple(supercell)

# Distinguish metal and oxygen atoms
metal_ids, oxygen_ids = [], []
for i, atom in enumerate(template):
    if atom.symbol == "O":
        oxygen_ids.append(i)
    else:
        metal_ids.append(i)
metal_atoms, oxygen_atoms = template[metal_ids], template[oxygen_ids]
sorted_ids = [0] * len(template)
for k, v in enumerate(metal_ids + oxygen_ids):
    sorted_ids[v] = k
bulk = (metal_atoms + oxygen_atoms)[sorted_ids]

# --- Generate population ---
input_pop = []
pop_size_per_valence = len(temperatures) * replicas_per_valence

for dopant_valence in dopant_valences:
    valence_pop = generate_population(
        template=bulk,
        pop_size=pop_size_per_valence,
        target_metal_species=target_metal_species,
        target_metal_composition=target_metal_composition,
        target_nonmetal_species=target_nonmetal_species,
        valences={dopant: dopant_valence, host_metal: 4, "O": -2},
    )
    for atoms in valence_pop:
        atoms.info = {"key_value_pairs": {}, "data": {}}
        atoms.info["key_value_pairs"]["dopant_valence"] = dopant_valence
        input_pop.append(atoms)

# --- Assign temperatures and replica indices ---
rep = 0
for xi in range(len(dopant_valences)):
    for T in temperatures:
        for _ in range(replicas_per_valence):
            input_pop[rep].info["key_value_pairs"]["temperature"] = T
            input_pop[rep].info["key_value_pairs"]["replica"] = rep
            input_pop[rep].info["key_value_pairs"]["emc_step"] = 0
            rep += 1

# --- Report ---
for idx, img in enumerate(input_pop):
    kvp = img.info["key_value_pairs"]
    print(f"Replica {idx} -> valence={kvp['dopant_valence']}, T={kvp['temperature']:.1f}")

out_fname = os.path.join(cwd, f"{system_name}_input_pop_unrelaxed.traj")
write(out_fname, input_pop)
print()
print(f"Written {len(input_pop)} replicas to {out_fname}")
