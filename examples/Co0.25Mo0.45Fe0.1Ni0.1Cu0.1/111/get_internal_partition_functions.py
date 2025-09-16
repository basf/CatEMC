from ase import units
from acat.settings import adsorbate_molecule
import numpy as np
import pickle


temperatures = [573, 673, 773, 873]
internal_data = {T: {} for T in temperatures}
degeneracies = {'N': 4, 'H': 2, 'NH': 2, 'NH2': 2, 'N2': 1, 'NH3': 1, 'H2': 1}
sigmas = {'N': 1, 'H': 1, 'NH': 1, 'NH2': 2, 'N2': 2, 'NH3': 3, 'H2': 2}
for temperature in temperatures:
    for s in degeneracies.keys():
        if s in ['N','H']:
            q_rot = 1
        elif s in ['NH','N2','H2']:
            atoms = adsorbate_molecule(s)
            inertias = atoms.get_moments_of_inertia() * units._amu / (10.0**10)**2
            inertia = max(inertias)  # should be two identical and one zero
            q_rot = 8 * np.pi**2 * inertia * units._k * temperature / sigmas[s] / units._hplanck**2
        elif s in ['NH2','NH3']:
            atoms = adsorbate_molecule(s)
            inertias = atoms.get_moments_of_inertia() * units._amu / (10.0**10)**2  # kg m^2
            q_rot = np.sqrt(np.pi * np.prod(inertias)) / sigmas[s]
            q_rot *= (8.0 * np.pi**2 * units._k * temperature / units._hplanck**2)**(3.0 / 2.0)
        internal_data[temperature][s] = q_rot * degeneracies[s]
print(internal_data)
with open(f'NH_internal_data.pkl', 'wb') as handle:
    pickle.dump(internal_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
