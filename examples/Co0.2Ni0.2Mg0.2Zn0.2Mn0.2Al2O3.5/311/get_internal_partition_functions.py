from ase import units
from acat.settings import adsorbate_molecule
import numpy as np
import pickle


temperatures = [623, 723, 823]
internal_data = {T: {} for T in temperatures}
degeneracies = {'C': 3, 'O': 3, 'H': 2, 'CO': 1, 'CO2': 1, 'OH': 2, 'OH2': 1, 'H2': 1, 'O2': 1, 'OOH': 2, 'O2H': 2, 'COH': 2, 'CHO': 2, 'CH': 2, 'CH2': 3, 'CH3': 2}
sigmas = {'C': 1, 'O': 1, 'H': 1, 'CO': 1, 'CO2': 2, 'OH': 1, 'OH2': 2, 'H2': 2, 'O2': 2, 'OOH': 1, 'O2H': 1, 'COH': 1, 'CHO': 1, 'CH': 1, 'CH2': 1, 'CH3': 3}
for temperature in temperatures:
    for s in degeneracies.keys():
        if s in ['C','O','H']:
            q_rot = 1
        elif s in ['CO','CO2','OH','H2','O2','CH']:
            atoms = adsorbate_molecule(s)
            inertias = atoms.get_moments_of_inertia() * units._amu / (10.0**10)**2
            inertia = max(inertias)  # should be two identical and one zero
            q_rot = 8 * np.pi**2 * inertia * units._k * temperature / sigmas[s] / units._hplanck**2
        elif s in ['OH2','O2H','OOH','COH','CHO','CH2','CH3']:
            atoms = adsorbate_molecule(s)
            inertias = atoms.get_moments_of_inertia() * units._amu / (10.0**10)**2  # kg m^2
            q_rot = np.sqrt(np.pi * np.prod(inertias)) / sigmas[s]
            q_rot *= (8.0 * np.pi**2 * units._k * temperature / units._hplanck**2)**(3.0 / 2.0)
        internal_data[temperature][s] = q_rot * degeneracies[s]
print(internal_data)
with open(f'COH_internal_data.pkl', 'wb') as handle:
    pickle.dump(internal_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
