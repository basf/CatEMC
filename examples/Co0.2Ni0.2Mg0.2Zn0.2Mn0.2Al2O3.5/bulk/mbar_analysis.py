from catemc.analysis import GEEMCAnalysis
from ase import units
import pickle

system_name = 'Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5_bulk'
dir_name = '/mnt/nfs/emc_tutorial/Co0.2Ni0.2Mg0.2Zn0.2Mn0.2Al2O3.5/bulk'
ana = GEEMCAnalysis(
        name=system_name,
        calculation_dir=dir_name,
        pop_size=48, 
        target_metal_species=['Co','Ni','Mg','Zn','Mn','Al'],
        trim_generation=100,
        temperature_key='temperature',
        replica_key='replica',
        acceptance_key='accepted')

ana.prepare_data(use_exist=True, pressure=1.01325*units.bar, raw_score_cutoff=-10000.)
ana.get_trajectory(replica=10)
ana.get_sro(starting_step=300, ending_step=1000, sample_stride=10, cutoff=3.5, use_energy_mask=True)
ana.get_dmu_estimates(starting_step=300, ending_step=1000, sample_stride=10, 
                      target_temperatures=[723, 800, 823, 873, 950, 1023, 1100, 1173],
                      volume_correction=True, wls_correction=True)

with open(f'{dir_name}/{system_name}_mbar_dmu_estimates.pkl', 'rb') as handle:
    mbar_dmu_estimates = pickle.load(handle)
for T, d in mbar_dmu_estimates.items():
    print(T)
    print({k: round(v, 2) for k, v in d.items()})
