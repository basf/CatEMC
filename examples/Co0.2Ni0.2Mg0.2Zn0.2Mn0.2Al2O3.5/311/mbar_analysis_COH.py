from catemc.analysis import GCEMCAnalysis
from ase.io import read, write
import numpy as np
import pickle

system_name = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5COH_311'
dir_name = '/mnt/nfs/spd_paper/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/311'
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
dmu_C = np.arange(0., 2.001, 0.05)
dmu_O = np.arange(-4., -2.999, 0.025)
dmu_H = np.arange(-1., 0.001, 0.1)
sim_temperatures = [623, 723, 823] 
grid = np.array(np.meshgrid(dmu_C, dmu_O, dmu_H, sim_temperatures)).T.reshape(-1, 4)
early_mu_C = -8.766
early_dmu_C = early_mu_C - E_C
early_mu_O = -7.811
early_dmu_O = early_mu_O - E_O2/2
early_cond = np.array([early_dmu_C, early_dmu_O, -0.5, 723])
equil_mu_C = -9.241
equil_dmu_C = equil_mu_C - E_C
equil_mu_O = -7.605
equil_dmu_O = equil_mu_O - E_O2/2
equil_cond = np.array([equil_dmu_C, equil_dmu_O, -0.5, 723])
reaction_conditions = np.vstack([grid, early_cond, equil_cond])
with open(f'/mnt/nfs/spd_paper/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5_bulk_mbar_dmu_estimates.pkl', 'rb') as f:
    tdmu_map = pickle.load(f)
template = read(f'{dir_name}/template_Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5_311_slab_mace_opt.traj')
z_surf = None

ana.get_structural_expectations(
        reaction_conditions=reaction_conditions,
        temperature_dmu_map=tdmu_map,
        element_column_map={'C': 0, 'O': 1, 'H': 2},
        element_eref_map={'C': 0., 'O': E_O2/2, 'H': E_H2/2},
        element_y2x_map={'C': 0., 'O': 3.5/3, 'H': 0.},
        z_surf=z_surf,
        starting_step=500, 
        ending_step=2000, 
        sample_stride=50,
        vacancy_species=['O'])

sim_dmu_C = np.arange(0., 2.001, 0.1)
sim_dmu_O = np.arange(-4., -2.999, 0.1)
sim_dmu_H = np.arange(-1., 0.001, 0.1)
sim_grid = np.array(np.meshgrid(sim_dmu_C, sim_dmu_O, sim_dmu_H, sim_temperatures)).T.reshape(-1, 4)
sim_reaction_conditions = np.vstack([sim_grid, early_cond, equil_cond])
ana.get_structural_fes(
        reaction_conditions=sim_reaction_conditions,
        temperature_dmu_map=tdmu_map,
        element_column_map={'C': 0, 'O': 1, 'H': 2},
        element_eref_map={'C': 0., 'O': E_O2/2, 'H': E_H2/2},
        element_y2x_map={'C': 0., 'O': 3.5/3, 'H': 0.},
        z_surf=z_surf,
        starting_step=500, 
        ending_step=2000, 
        sample_stride=50,
        sample_condition_indices=[-2,-1],
        vacancy_species=['O'],
        bin_width=0.05)
