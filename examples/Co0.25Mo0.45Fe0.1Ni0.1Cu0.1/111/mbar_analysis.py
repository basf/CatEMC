from catemc.analysis import GCEMCAnalysis
from ase.io import read, write
import numpy as np
import pickle

system_name = 'Co0.25Mo0.45Fe0.1Ni0.1Cu0.1NH_111'
dir_name = '/emc/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/111'
ana = GCEMCAnalysis(
        name=system_name,
        calculation_dir=dir_name,
        pop_size=56,
        target_metal_species=['Co','Mo','Fe','Ni','Cu'],
        target_metal_composition={'Co':0.25,'Mo':0.45,'Fe':0.1,'Ni':0.1,'Cu':0.1},
        adsorbate_species=['N','H','NH','NH2','N2'],
        trim_generation=100,
        temperature_key='temperature',
        replica_key='replica',
        acceptance_key='accepted')

ana.prepare_data(use_exist=True, raw_score_cutoff=-10000.)
ana.get_trajectory(replica=10)

E_N2 = -16.293 + 0.155  # Etot incl. ZPE RPBE 
E_H2 = -6.984 + 0.269   # Etot incl. ZPE RPBE
dmu_N = np.arange(-1., 0.001, 0.025)
dmu_H = np.arange(-1., 0.001, 0.025)
sim_temperatures = [573, 673, 773, 873]
target_temperature = 800
grid = np.array(np.meshgrid(dmu_N, dmu_H, sim_temperatures+[target_temperature])).T.reshape(-1, 3)
early_mu_N = -8.227
early_dmu_N = early_mu_N - E_N2/2
early_mu_H = -4.065
early_dmu_H = early_mu_H - E_H2/2
early_cond = np.array([early_dmu_N, early_dmu_H, target_temperature])
equil_mu_N = -8.993
equil_dmu_N = equil_mu_N - E_N2/2
equil_mu_H = -3.985 
equil_dmu_H = equil_mu_H - E_H2/2
equil_cond = np.array([equil_dmu_N, equil_dmu_H, target_temperature])
reaction_conditions = np.vstack([grid, early_cond, equil_cond])
with open(f'/emc/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1/bulk/Co0.25Mo0.45Fe0.1Ni0.1Cu0.1_bulk_mbar_dmu_estimates.pkl', 'rb') as f:
    tdmu_map = pickle.load(f)

ana.get_structural_expectations(
        z_surf=22.,
        reaction_conditions=reaction_conditions,
        temperature_dmu_map=tdmu_map,
        element_column_map={'N': 0, 'H': 1},
        element_eref_map={'N': E_N2/2, 'H': E_H2/2},
        element_y2x_map={'N': 0., 'H': 0.},
        starting_step=500, 
        ending_step=2000, 
        sample_stride=50,
        vacancy_indices=None)

sim_dmu_N = np.arange(-1., 0.001, 0.05)
sim_dmu_H = np.arange(-1., 0.001, 0.05)
sim_grid = np.array(np.meshgrid(sim_dmu_N, sim_dmu_H, sim_temperatures)).T.reshape(-1, 3)
sim_reaction_conditions = np.vstack([sim_grid, early_cond, equil_cond])
ana.get_structural_fes(
        z_surf=22.,
        reaction_conditions=sim_reaction_conditions,
        temperature_dmu_map=tdmu_map,
        element_column_map={'N': 0, 'H': 1},
        element_eref_map={'N': E_N2/2, 'H': E_H2/2},
        element_y2x_map={'N': 0., 'H': 0.},
        starting_step=500, 
        ending_step=2000, 
        sample_stride=50,
        sample_condition_indices=[-2,-1],
        bin_width=0.05)
