from catemc.analysis import GCEMCAnalysis
from ase.io import read, write
import numpy as np
import pickle

system_name = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5COH_311'
dir_name = '/mnt/nfs/emc_tutorial/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/311'
ana = GCEMCAnalysis(
        name=system_name,
        calculation_dir=dir_name,
        pop_size=42, 
        target_metal_species=['Co','Ni','Mg','Zn','Mn','Al'],
        target_metal_composition={'Co':0.2,'Ni':0.2,'Mg':0.2,'Zn':0.2,'Mn':0.2,'Al':2},
        adsorbate_species=['O','H','OH','CO','CO2','CH','CH2','CH3'],
        trim_generation=100,
        temperature_key='temperature',
        replica_key='replica',
        acceptance_key='accepted')

ana.prepare_data(use_exist=True, raw_score_cutoff=-10000.)
ana.get_trajectory(replica=10)

E_C = -9.944    # Etot per atom graphite PBE
E_O2 = -8.768 + 0.101   # Etot incl. ZPE PBE
E_H2 = -6.773 + 0.267   # Etot incl. ZPE PBE
dmu_C = np.arange(0., 2.001, 0.1)
dmu_O = np.arange(-4., -2.999, 0.1)
dmu_H = np.arange(-1.0, 0.001, 0.1)
sim_temperatures = [723,]
reaction_conditions = np.array(np.meshgrid(dmu_C, dmu_O, dmu_H, sim_temperatures)).T.reshape(-1, 4)
with open(f'/mnt/nfs/emc_tutorial/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5_bulk_wls_dmu_estimates.pkl', 'rb') as f:
    tdmu_map = pickle.load(f)
template = read(f'{dir_name}/template_Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O4_311_slab_mace_opt.traj')
vac_indices = [a.index for a in template if a.symbol in ['O','X']]

ana.get_structural_expectations(
        z_surf=24.,
        reaction_conditions=reaction_conditions,
        temperature_dmu_map=tdmu_map,
        element_column_map={'C': 0, 'O': 1, 'H': 2},
        element_eref_map={'C': E_C, 'O': E_O2/2, 'H': E_H2/2},
        element_y2x_map={'C': 0., 'O': 4/3, 'H': 0.},
        starting_step=500, 
        ending_step=2000, 
        sample_stride=50,
        vacancy_indices=vac_indices)
