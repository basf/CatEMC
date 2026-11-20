"""CatEMC quickstart: Gibbs-ensemble EMC of a small CuNiPdPtAu alloy.

The script generates the initial population, runs the simulation and
estimates the chemical-potential differences between the metals. It uses
the EMT calculator of ASE and takes about a minute on a laptop.

    python quickstart.py

The same study from the command line is in config.yaml:

    catemc generate config.yaml
    catemc run      config.yaml
    catemc analyze  config.yaml
"""
from ase import units
from ase.build import bulk
from catemc.generate import generate_population
from catemc.geemc import GEEMC
from catemc.analysis import GEEMCAnalysis

NAME = 'CuNiPdPtAu_bulk'
METALS = {'Cu': 0.25, 'Ni': 0.25, 'Pd': 0.2, 'Pt': 0.2, 'Au': 0.1}
TEMPERATURES = [600, 700, 800, 900]
REPLICAS_PER_TEMPERATURE = 2
POP_SIZE = len(TEMPERATURES) * REPLICAS_PER_TEMPERATURE


def create_calculators():
    """Called once in every worker process. Must be defined at module level."""
    from ase.calculators.emt import EMT
    return [EMT()]


def main():
    # 1) Structure generation: random decorations of a 32-atom fcc cell
    template = bulk('Cu', 'fcc', a=3.8, cubic=True) * (2, 2, 2)
    input_pop = generate_population(
        template=template, pop_size=POP_SIZE,
        target_metal_species=list(METALS),
        target_metal_composition=METALS, seed=1)
    for replica, atoms in enumerate(input_pop):
        temperature = TEMPERATURES[replica // REPLICAS_PER_TEMPERATURE]
        atoms.info = {'key_value_pairs': {'temperature': temperature,
                                          'replica': replica,
                                          'emc_step': 0},
                      'data': {}}

    # 2) Gibbs-ensemble EMC
    emc = GEEMC(
        NAME, calculation_dir='.', input_pop=input_pop,
        calculator_factory=create_calculators,
        fitness_kwargs=dict(fmax_list=[0.1],
                            md_timestep=2 * units.fs,
                            md_taut=100 * units.fs,
                            md_steps=20,
                            md_write_interval=10,
                            neighbor_cutoff=3.5,
                            optimizer='LBFGS'),
        metal_element_pool=list(METALS),
        metal_interswap_rate=0.8,
        intraswap_rate=0.1,
        crossover_rate=0.1,
        md_generation=10,
        exchange_generation=1,
        bias_starting_step=10,
        trim_generation=20,
        procs_per_device=2)
    emc.run(total_steps=100)

    # 3) Post-simulation analysis
    ana = GEEMCAnalysis(name=NAME, calculation_dir='.', pop_size=POP_SIZE,
                        target_metal_species=list(METALS),
                        trim_generation=20)
    ana.prepare_data()
    ana.get_sro(starting_step=20, cutoff=3.5)
    dmu = ana.get_dmu_estimates(starting_step=20,
                                target_temperatures=[600, 650, 700, 800, 900])
    for temperature, values in dmu.items():
        print(temperature,
              {pair: round(float(value), 3) for pair, value in values.items()})


if __name__ == '__main__':
    main()
