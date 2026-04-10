from catemc.analysis import GCEMCAnalysis
from ase.io import read, write
import numpy as np
import pickle

system_name = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5OH_311'
dir_name = '/emc/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/311'
ana = GCEMCAnalysis(
        name=system_name,
        calculation_dir=dir_name,
        pop_size=14,
        target_metal_species=['Co','Ni','Mg','Zn','Mn','Al'],
        target_metal_composition={'Co':0.2,'Ni':0.2,'Mg':0.2,'Zn':0.2,'Mn':0.2,'Al':2},
        adsorbate_species=['O','H','OH'],
        trim_generation=100,
        temperature_key='temperature',
        replica_key='replica',
        acceptance_key='accepted')

ana.prepare_data(use_exist=True, raw_score_cutoff=-10000.)
ana.get_trajectory(replica=10)

E_O2 = -8.768 + 0.101   # Etot incl. ZPE PBE
E_H2 = -6.773 + 0.267   # Etot incl. ZPE PBE
dmu_O = np.arange(-5., -2.999, 0.02)
dmu_H = np.arange(-1., 0.001, 0.01)
sim_temperatures = [1023,]
grid = np.array(np.meshgrid(dmu_O, dmu_H, sim_temperatures)).T.reshape(-1, 3)
target_T = 1023
target_dmu_O = -8.821 - E_O2 / 2 
target_dmu_H = -3.981 - E_H2 / 2
target_cond = np.array([target_dmu_O, target_dmu_H, target_T])
reaction_conditions = np.vstack([grid, target_cond])
with open(f'/emc/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5_bulk_mbar_dmu_estimates.pkl', 'rb') as f:
    tdmu_map = pickle.load(f)
template = read(f'{dir_name}/template_Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5_311_slab_mace_opt.traj')
z_surf = None

ana.get_structural_expectations(
        reaction_conditions=reaction_conditions,
        temperature_dmu_map=tdmu_map,
        element_column_map={'O': 0, 'H': 1},
        element_eref_map={'O': E_O2/2, 'H': E_H2/2},
        element_y2x_map={'O': 3.5/3, 'H': 0.},
        z_surf=z_surf,
        starting_step=500, 
        ending_step=4000, 
        sample_stride=50,
        vacancy_species=['O'])

sim_dmu_O = np.arange(-5., -2.999, 0.1)
sim_dmu_H = np.arange(-1., 0.001, 0.05)
sim_grid = np.array(np.meshgrid(sim_dmu_O, sim_dmu_H, sim_temperatures)).T.reshape(-1, 3)
sim_reaction_conditions = np.vstack([sim_grid, target_cond])
ana.get_structural_fes(
        reaction_conditions=sim_reaction_conditions,
        temperature_dmu_map=tdmu_map,
        element_column_map={'O': 0, 'H': 1},
        element_eref_map={'O': E_O2/2, 'H': E_H2/2},
        element_y2x_map={'O': 3.5/3, 'H': 0.},
        z_surf=z_surf,
        starting_step=500, 
        ending_step=4000, 
        sample_stride=50,
        sample_condition_indices=[-1],
        vacancy_species=['O'],
        bin_width=0.05)
