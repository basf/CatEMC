#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Translate a CatEMC YAML configuration into CatEMC objects.

This module is the engine behind the ``catemc`` command-line interface
(:mod:`catemc.cli`). It contains no sampling logic. It only builds the
arguments of :func:`catemc.generate.generate_population`,
:class:`catemc.emc.EMC`, :class:`catemc.geemc.GEEMC`,
:class:`catemc.gcemc.GCEMC` and the analysis classes from one YAML file, so
that a CLI run and the equivalent Python script do exactly the same thing.

Units used in the YAML file: temperatures in K, energies and chemical
potentials in eV, lengths in angstrom, times in fs and pressures in bar.
"""
from ase import data as ase_data
from ase import units
from ase.io import read, write
from collections import Counter
import difflib
import importlib
import importlib.util
import inspect
import math
import numpy as np
import os
import random
import re


ENSEMBLES = {
    'canonical': 'canonical', 'emc': 'canonical', 'nvt': 'canonical',
    'gibbs': 'gibbs', 'geemc': 'gibbs', 'ge': 'gibbs',
    'grand_canonical': 'grand_canonical', 'gcemc': 'grand_canonical',
    'gc': 'grand_canonical', 'egc': 'grand_canonical',
}

SECTIONS = ('name', 'calculation_dir', 'ensemble', 'composition', 'regions',
            'population', 'calculators', 'fitness', 'emc', 'run',
            'adsorbates', 'chemical_potentials', 'analysis')

# Elements assigned to the non-metal sublattice when the composition is
# given as one formula string. Use the explicit metals/nonmetals form for
# anything else (e.g. borides or silicides).
NONMETALS = {'H', 'He', 'C', 'N', 'O', 'F', 'Ne', 'P', 'S', 'Cl', 'Ar',
             'Se', 'Br', 'Kr', 'I', 'Xe'}

CALCULATOR_ALIASES = {
    'repulsive': 'catemc.repulsive.RepulsivePotential',
    'emt': 'ase.calculators.emt.EMT',
    'mace': 'mace.calculators.mace.MACECalculator',
    'mace_mp': 'mace.calculators.mace_mp',
}

# Fitness settings given in fs (or 1/fs, 1/bar) in the YAML file and
# converted to ASE units here.
FITNESS_UNITS = {
    'md_timestep': units.fs,
    'md_taut': units.fs,
    'md_taup': units.fs,
    'md_friction': 1. / units.fs,
    'compressibility': 1. / units.bar,
}

# Defaults of the built-in fitness functions that have no default in the
# Python API. They are the settings used in the CatEMC paper.
FITNESS_DEFAULTS = {
    'md_timestep': 1.0,
    'md_taut': 500.0,
    'md_steps': 400,
    'md_write_interval': 10,
}

# Constructor arguments that are Python objects and can therefore not be
# written in the ``emc`` section of the YAML file.
OBJECT_KEYS = {'name', 'calculation_dir', 'input_pop', 'calculator_factory',
               'calculator_factory_kwargs', 'fitness_function',
               'fitness_kwargs', 'adsorption_sites'}


class ConfigError(ValueError):
    """Raised for an invalid CatEMC configuration file."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _suggest(key, options):
    match = difflib.get_close_matches(str(key), [str(o) for o in options], n=1)
    return f" Did you mean '{match[0]}'?" if match else ''


def _check_keys(given, allowed, where):
    for key in given:
        if key not in allowed:
            raise ConfigError(
                f"Unknown key '{key}' in {where}.{_suggest(key, allowed)} "
                f"Allowed keys: {', '.join(sorted(map(str, allowed)))}.")


def _expand(value):
    """Expand ``~`` and environment variables in every string of a config."""
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def import_object(path):
    """Import ``package.module.Object`` (or ``package.module:Object``).

    Attribute chains such as ``package.module.Class.from_file`` are resolved
    as well, so class methods can be used as calculator constructors.
    """
    if ':' in path:
        module_name, attrs = path.split(':', 1)
        obj = importlib.import_module(module_name)
        for attr in attrs.split('.'):
            obj = getattr(obj, attr)
        return obj
    parts = path.split('.')
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module('.'.join(parts[:i]))
        except ModuleNotFoundError as err:
            # Only skip when this very module is the one that is missing.
            if err.name is None or not '.'.join(parts[:i]).startswith(err.name):
                raise
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"Cannot import '{path}'")


def create_calculators(stages=None, factory=None, factory_kwargs=None):
    """Calculator factory used by the CLI.

    It is defined at module level because every persistent worker imports
    and calls it once (see :class:`catemc.worker_pool.WorkerPool`).

    Parameters
    ----------
    stages : list of dict, optional
        One entry per calculator. ``calculator`` is the import path of the
        calculator class (or one of the short names ``repulsive``, ``emt``,
        ``mace``, ``mace_mp``), ``args`` an optional list of positional arguments and all
        remaining entries are keyword arguments.
    factory : str, optional
        ``path/to/file.py:function`` or ``package.module:function`` of a
        user-defined factory returning one calculator or a list of them.
    factory_kwargs : dict, optional
        Keyword arguments passed to the user-defined factory.
    """
    if factory is not None:
        target, _, func_name = factory.rpartition(':')
        if target.endswith('.py'):
            spec = importlib.util.spec_from_file_location(
                '_catemc_calculator_factory', target)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(target)
        return getattr(module, func_name)(**(factory_kwargs or {}))

    calculators = []
    for stage in stages or []:
        kwargs = dict(stage)
        path = kwargs.pop('calculator')
        args = kwargs.pop('args', [])
        cls = import_object(CALCULATOR_ALIASES.get(path, path))
        calculators.append(cls(*args, **kwargs))
    return calculators


def parse_formula(formula):
    """``'Co0.2Ni0.2Al2O3.5'`` -> ``{'Co': 0.2, 'Ni': 0.2, 'Al': 2.0, 'O': 3.5}``.

    The order of the elements is preserved. ``X`` denotes a vacancy.
    """
    tokens = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*(?:[eE][-+]?\d+)?)', formula)
    if not tokens or ''.join(s + n for s, n in tokens) != formula.replace(' ', ''):
        raise ConfigError(f"Cannot parse the composition '{formula}'.")
    composition = {}
    for symbol, number in tokens:
        if symbol != 'X' and symbol not in ase_data.atomic_numbers:
            raise ConfigError(f"Unknown element '{symbol}' in '{formula}'.")
        composition[symbol] = composition.get(symbol, 0.) + (
            float(number) if number else 1.)
    return composition


def parse_composition(spec):
    """Return ``(metals, nonmetals)`` from the ``composition`` entry.

    ``spec`` is one formula string, which is split into the metal and
    non-metal sublattices automatically, or a mapping with the keys
    ``metals`` and (optionally) ``nonmetals``. Each of these is again a
    formula string or an ``{element: amount}`` mapping. ``nonmetals`` is
    ``None`` when the material has no non-metal sublattice.
    """
    def one(value, where):
        if value is None:
            return None
        if isinstance(value, str):
            return parse_formula(value)
        if isinstance(value, dict):
            out = {}
            for symbol, amount in value.items():
                symbol = str(symbol)
                if symbol != 'X' and symbol not in ase_data.atomic_numbers:
                    raise ConfigError(f"Unknown element '{symbol}' in {where}.")
                out[symbol] = float(amount)
            return out
        raise ConfigError(f'{where} must be a formula or an element mapping.')

    if spec is None:
        raise ConfigError("The 'composition' entry is required.")
    if isinstance(spec, str):
        full = parse_formula(spec)
        nonmetals = {k: v for k, v in full.items() if k in NONMETALS}
        metals = {k: v for k, v in full.items() if k not in NONMETALS and k != 'X'}
        if 'X' in full:
            (nonmetals if nonmetals else metals)['X'] = full['X']
        nonmetals = nonmetals or None
    else:
        _check_keys(spec, ('metals', 'nonmetals'), 'composition')
        metals = one(spec.get('metals'), 'composition.metals')
        nonmetals = one(spec.get('nonmetals'), 'composition.nonmetals')
    if not metals:
        raise ConfigError('The composition must contain at least one metal.')
    return metals, nonmetals


def select(atoms, spec, where='selection'):
    """Return the sorted atom indices described by ``spec``.

    ``spec`` is ``'all'``, ``'none'``/``None``, a list of indices or a mapping
    whose criteria are combined with a logical AND:

    ``z_below`` / ``z_above``
        atoms with ``z < value`` / ``z >= value`` (angstrom);
    ``bottom_layers`` / ``top_layers``
        the lowest / highest N atomic layers along z;
    ``symbols``
        atoms of the listed chemical symbols;
    ``indices``
        an explicit list of atom indices.
    """
    n = len(atoms)
    if spec is None or spec == 'none':
        return []
    if spec == 'all':
        return list(range(n))
    if isinstance(spec, (list, tuple)):
        return sorted(int(i) for i in spec)
    if not isinstance(spec, dict):
        raise ConfigError(f'Cannot interpret {where}: {spec!r}')
    allowed = ('z_below', 'z_above', 'bottom_layers', 'top_layers',
               'layer_tolerance', 'symbols', 'indices')
    _check_keys(spec, allowed, where)
    mask = np.ones(n, dtype=bool)
    z = atoms.positions[:, 2]
    if spec.get('z_below') is not None:
        mask &= z < float(spec['z_below'])
    if spec.get('z_above') is not None:
        mask &= z >= float(spec['z_above'])
    if spec.get('bottom_layers') is not None or spec.get('top_layers') is not None:
        from ase.geometry import get_layers
        layers = get_layers(atoms, (0, 0, 1),
                            tolerance=float(spec.get('layer_tolerance', 0.3)))[0]
        if spec.get('bottom_layers') is not None:
            mask &= layers < int(spec['bottom_layers'])
        if spec.get('top_layers') is not None:
            mask &= layers > layers.max() - int(spec['top_layers'])
    if spec.get('symbols') is not None:
        mask &= np.isin(atoms.get_chemical_symbols(), _as_list(spec['symbols']))
    if spec.get('indices') is not None:
        chosen = np.zeros(n, dtype=bool)
        chosen[[int(i) for i in spec['indices']]] = True
        mask &= chosen
    return np.where(mask)[0].tolist()


def _axis(spec, where):
    """Chemical-potential axis from ``values`` or ``range`` + ``step``."""
    if spec.get('values') is not None:
        if spec.get('range') is not None or spec.get('step') is not None:
            raise ConfigError(f"{where} needs either 'values' or 'range' and "
                              "'step', not both.")
        return np.array(_as_list(spec['values']), dtype=float)
    if spec.get('range') is None or spec.get('step') is None:
        raise ConfigError(f"{where} needs 'values' or both 'range' and 'step'.")
    lo, hi = (float(v) for v in spec['range'])
    step = float(spec['step'])
    if step <= 0 or hi < lo:
        raise ConfigError(f"{where} needs a positive 'step' and a 'range' [low, high].")
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 10)


def _mu_spec(par, where, allowed=('reference', 'range', 'step', 'values')):
    """Check one ``{reference, range, step}`` or ``{reference, values}`` entry."""
    if not isinstance(par, dict):
        raise ConfigError(f'{where} must be a mapping such as '
                          '{reference: -8.2425, range: [-1.0, 0.0], step: 0.05}.')
    _check_keys(par, allowed, where)
    if par.get('reference') is None:
        raise ConfigError(f"{where} needs 'reference', the reference energy "
                          'to which range and step (or values) are added.')
    return par


def make_grid(axes, temperatures=None):
    """Reaction-condition grid with one row per state.

    The chemical-potential columns follow the order of ``axes`` and the
    temperature, when given, is the last column.
    """
    arrays = list(axes)
    if temperatures is not None:
        arrays.append(np.array(temperatures, dtype=float))
    return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))


def internal_partition_functions(species, temperatures):
    """Internal partition functions of rigid adsorbate fragments.

    ``Q_int = g0 * Q_rot`` with the electronic ground-state degeneracy ``g0``
    and the classical rigid-rotor partition function ``Q_rot`` (1 for an
    atom, ``8 pi^2 I k T / (sigma h^2)`` for a linear fragment and
    ``sqrt(pi Ix Iy Iz) / sigma * (8 pi^2 k T / h^2)^(3/2)`` otherwise).

    Parameters
    ----------
    species : dict
        ``{name: {'degeneracy': g0, 'symmetry_number': sigma}}`` with ACAT
        fragment names. A plain number is used directly as ``Q_int`` at all
        temperatures, and a ``{temperature: Q_int}`` mapping gives the value
        at each temperature.
    temperatures : sequence of float
        Temperatures in K.

    Returns
    -------
    dict
        ``{temperature: {name: Q_int}}``, the ``temperature_species_internal_map``
        of :class:`catemc.gcemc.GCEMC`.
    """
    from acat.settings import adsorbate_molecule

    where = 'adsorbates.internal_partition_functions'
    internal = {T: {} for T in temperatures}
    for name, par in species.items():
        if isinstance(par, (int, float)):
            for T in temperatures:
                internal[T][name] = float(par)
            continue
        if not isinstance(par, dict):
            raise ConfigError(f'{where}.{name} must be a number or a mapping.')
        if par and all(isinstance(k, (int, float)) for k in par):
            for T in temperatures:
                value = _temperature_value(par, T)
                if value is None:
                    raise ConfigError(f'{where}.{name} has no value at {T} K.')
                internal[T][name] = float(value)
            continue
        _check_keys(par, ('degeneracy', 'symmetry_number'), f'{where}.{name}')
        g0 = float(par.get('degeneracy', 1))
        sigma = float(par.get('symmetry_number', 1))
        molecule = adsorbate_molecule(name)
        if molecule is None:
            raise ConfigError(
                f"'{name}' is not a known ACAT fragment. Give its internal "
                'partition function as a number, or as a mapping '
                '{temperature: value}.')
        if len(molecule) > 1:
            inertias = molecule.get_moments_of_inertia() * units._amu / 1e20
            linear = inertias.min() < 1e-6 * inertias.max()
        for T in temperatures:
            if len(molecule) == 1:
                q_rot = 1.
            elif linear:
                q_rot = (8 * np.pi**2 * inertias.max() * units._k * T
                         / sigma / units._hplanck**2)
            else:
                q_rot = np.sqrt(np.pi * np.prod(inertias)) / sigma
                q_rot *= (8 * np.pi**2 * units._k * T / units._hplanck**2)**1.5
            internal[T][name] = float(q_rot * g0)
    return internal


def _temperature_value(mapping, T):
    """Value of a ``{temperature: value}`` mapping at temperature ``T``."""
    for key, value in mapping.items():
        if isinstance(key, (int, float)) and math.isclose(float(key), float(T)):
            return value
    return None


# ---------------------------------------------------------------------------
# The configuration object
# ---------------------------------------------------------------------------

class Config(object):

    """A parsed CatEMC YAML configuration.

    Parameters
    ----------
    path : str
        Path of the YAML file. Relative paths inside the file are resolved
        with respect to the folder that contains it.
    overrides : list of str, optional
        ``section.key=value`` strings that replace entries of the file, e.g.
        ``['run.total_steps=200', 'emc.gpu_devices=[0,1]']``.
    """

    def __init__(self, path, overrides=None):
        import yaml
        self.path = os.path.abspath(path)
        self.root = os.path.dirname(self.path)
        with open(self.path) as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ConfigError('The configuration file must contain a mapping.')
        for item in overrides or []:
            key, sep, value = item.partition('=')
            if not sep:
                raise ConfigError(f"Override '{item}' is not of the form key=value.")
            node = raw
            parts = key.strip().split('.')
            for part in parts[:-1]:
                if not isinstance(node.get(part), dict):
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = yaml.safe_load(value)
        _check_keys(raw, SECTIONS, 'the configuration file')
        self.raw = _expand(raw)

        if not self.raw.get('name'):
            raise ConfigError("The 'name' entry (system name) is required.")
        self.name = str(self.raw['name'])
        self.directory = self.resolve(self.raw.get('calculation_dir', '.'))
        ensemble = str(self.raw.get('ensemble', '')).lower().replace('-', '_')
        if ensemble not in ENSEMBLES:
            raise ConfigError(
                "'ensemble' must be one of canonical, gibbs or grand_canonical."
                + _suggest(ensemble, ENSEMBLES))
        self.ensemble = ENSEMBLES[ensemble]
        self.metals, self.nonmetals = parse_composition(self.raw.get('composition'))
        self._template = None

    # -- generic accessors --------------------------------------------------

    def section(self, key):
        value = self.raw.get(key)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ConfigError(f"The '{key}' section must be a mapping.")
        return value

    def resolve(self, path):
        """Absolute path of ``path`` relative to the configuration file."""
        return os.path.normpath(os.path.join(self.root, str(path)))

    def output(self, suffix):
        """Path of ``{name}{suffix}`` in the calculation directory."""
        return os.path.join(self.directory, self.name + suffix)

    @property
    def is_gc(self):
        return self.ensemble == 'grand_canonical'

    @property
    def metal_pool(self):
        return list(self.metals)

    @property
    def nonmetal_pool(self):
        return None if self.nonmetals is None else list(self.nonmetals)

    @property
    def temperatures(self):
        temperatures = _as_list(self.section('population').get('temperatures'))
        if not temperatures:
            raise ConfigError("'population.temperatures' is required.")
        return temperatures

    @property
    def pop_size(self):
        pop = self.section('population')
        return len(self.temperatures) * int(pop.get('replicas_per_temperature', 1))

    @property
    def population_path(self):
        path = self.section('population').get('input_pop')
        return self.output('_input_pop.traj') if path is None else self.resolve(path)

    # -- template and regions -----------------------------------------------

    def get_template(self):
        """Template structure: the file ``population.template``, read with ASE.

        The file can be in any format that ASE reads. If it holds several
        structures, the last one is used.
        """
        if self._template is not None:
            return self._template
        path = self.section('population').get('template')
        if path is None:
            raise ConfigError(
                "'population.template' is required: the file of the template "
                'structure, in any format that ASE can read.')
        if not isinstance(path, str):
            raise ConfigError(
                "'population.template' must be the name of a structure file "
                'that ASE can read.')
        path = self.resolve(path)
        if not os.path.isfile(path):
            raise ConfigError(f'The template file {path} was not found.')
        atoms = read(path)
        atoms.info = {}
        atoms.calc = None
        if not any(atom.symbol in self.metal_pool for atom in atoms):
            raise ConfigError(
                f'The template {path} contains none of the metals in the '
                'composition. Its metal sites must hold metals of the composition.')
        self._template = atoms
        return atoms

    def get_regions(self, template):
        """Index lists derived from the ``regions`` section."""
        spec = self.section('regions')
        _check_keys(spec, ('fix_indices', 'mutable_metal_indices',
                           'mutable_nonmetal_indices', 'slab_indices'), 'regions')
        symbols = template.get_chemical_symbols()
        fixed = select(template, spec.get('fix_indices'), 'regions.fix_indices')
        free = [i for i in range(len(template)) if i not in set(fixed)]
        regions = {
            'fix_indices': fixed or None,
            'mutable_metal_indices': None,
            'mutable_nonmetal_indices': None,
            'free_indices': free,
        }
        if spec.get('mutable_metal_indices') is not None:
            regions['mutable_metal_indices'] = select(
                template, spec['mutable_metal_indices'], 'regions.mutable_metal_indices')
        elif fixed or self.is_gc:
            regions['mutable_metal_indices'] = [
                i for i in free if symbols[i] in self.metal_pool]
        if self.nonmetal_pool is not None:
            if spec.get('mutable_nonmetal_indices') is not None:
                regions['mutable_nonmetal_indices'] = select(
                    template, spec['mutable_nonmetal_indices'],
                    'regions.mutable_nonmetal_indices')
            elif fixed or self.is_gc:
                regions['mutable_nonmetal_indices'] = [
                    i for i in free if symbols[i] in self.nonmetal_pool]
        if self.is_gc:
            regions['slab_indices'] = select(
                template, spec.get('slab_indices', 'all'), 'regions.slab_indices')
        return regions

    # -- calculators and fitness --------------------------------------------

    def get_calculator_factory_kwargs(self):
        """Keyword arguments of :func:`create_calculators` and the fmax list."""
        spec = self.raw.get('calculators')
        if not spec:
            raise ConfigError("The 'calculators' section is required.")
        if isinstance(spec, dict):
            _check_keys(spec, ('calculator_factory', 'calculator_factory_kwargs',
                               'fmax'), 'calculators')
            if spec.get('calculator_factory') is None:
                raise ConfigError("'calculators' needs 'calculator_factory', or "
                                  'a list with one entry per calculator.')
            factory = str(spec['calculator_factory'])
            target, _, func = factory.rpartition(':')
            if not target:
                raise ConfigError("'calculators.calculator_factory' must look like "
                                  "'file.py:function' or 'package.module:function'.")
            if target.endswith('.py'):
                factory = f'{self.resolve(target)}:{func}'
            return ({'factory': factory,
                     'factory_kwargs': spec.get('calculator_factory_kwargs') or {}},
                    _as_list(spec.get('fmax')))
        stages, fmax_list = [], []
        for i, stage in enumerate(_as_list(spec)):
            if not isinstance(stage, dict) or 'calculator' not in stage:
                raise ConfigError(f"calculators[{i}] needs a 'calculator' entry.")
            stage = dict(stage)
            fmax_list.append(stage.pop('fmax', None))
            # Arguments written as ./file or ../file are paths relative to
            # the configuration file.
            stages.append({
                k: self.resolve(v) if isinstance(v, str) and k != 'calculator'
                and v.startswith(('./', '../')) else v for k, v in stage.items()})
        return {'stages': stages}, fmax_list

    def get_fitness_kwargs(self, fitness_function, fmax_list):
        spec = dict(self.section('fitness'))
        allowed = [p for p in inspect.signature(fitness_function).parameters
                   if p not in ('atoms', 'calculator_list', 'run_md', '_emc_context')]
        _check_keys(spec, allowed, 'fitness')
        kwargs = dict(FITNESS_DEFAULTS)
        kwargs.update(spec)
        kwargs.setdefault('fmax_list', fmax_list)
        if not kwargs['fmax_list']:
            raise ConfigError("Give 'fmax' for the calculators or 'fitness.fmax_list'.")
        for key, unit in FITNESS_UNITS.items():
            if kwargs.get(key) is not None:
                kwargs[key] = float(kwargs[key]) * unit
        return kwargs

    # -- grand-canonical quantities -----------------------------------------

    def get_mu_elements(self):
        """Settings of the non-metal reservoir elements, in column order."""
        spec = self.section('chemical_potentials')
        _check_keys(spec, ('metals', 'nonmetals'), 'chemical_potentials')
        elements = spec.get('nonmetals')
        if not elements:
            raise ConfigError("'chemical_potentials.nonmetals' is required for "
                              'a grand-canonical simulation.')
        if not isinstance(elements, dict):
            raise ConfigError("'chemical_potentials.nonmetals' must map each "
                              'element to its settings.')
        for element, par in elements.items():
            _mu_spec(par, f'chemical_potentials.nonmetals.{element}',
                     allowed=('reference', 'range', 'step', 'values', 'y2x'))
        return {str(k): dict(v) for k, v in elements.items()}

    def get_mu_axes(self):
        """Chemical-potential differences of the simulation grid, per element."""
        return [_axis(par, f'chemical_potentials.nonmetals.{element}')
                for element, par in self.get_mu_elements().items()]

    def get_element_maps(self):
        """``element_column_map``, ``element_eref_map`` and ``element_y2x_map``."""
        elements = self.get_mu_elements()
        n_metal = sum(v for k, v in self.metals.items() if k != 'X')
        column, eref, y2x = {}, {}, {}
        for i, (element, par) in enumerate(elements.items()):
            column[element] = i
            eref[element] = float(par['reference'])
            if par.get('y2x') is not None:
                y2x[element] = float(par['y2x'])
            else:
                y2x[element] = (self.nonmetals or {}).get(element, 0.) / n_metal
        return column, eref, y2x

    def get_metal_dmu_map(self, temperatures):
        """Metal chemical-potential differences ``{T: {'A2B': mu_B - mu_A}}``.

        They are built from ``chemical_potentials.metals``, which gives the
        chemical potential of every metal as one number for all temperatures
        or as a ``{temperature: value}`` mapping. Only the differences between
        the metals enter, so any metal can serve as the zero.
        """
        spec = self.section('chemical_potentials').get('metals')
        if spec is None:
            return None
        where = 'chemical_potentials.metals'
        if not isinstance(spec, dict):
            raise ConfigError(
                f"'{where}' must map every metal to its chemical potential: one "
                'number, or a mapping {temperature: value}.')
        metals = [m for m in self.metal_pool if m != 'X']
        _check_keys(spec, metals, where)
        missing = [m for m in metals if m not in spec]
        if missing:
            raise ConfigError(f'{where} misses {", ".join(missing)}.')
        tdmu = {}
        for T in temperatures:
            mu = {}
            for m in metals:
                value = spec[m]
                if isinstance(value, dict):
                    value = _temperature_value(value, T)
                if value is None or not np.isfinite(float(value)):
                    raise ConfigError(
                        f'{where}.{m} has no value at {T} K. The Gibbs-ensemble '
                        'analysis of the bulk (analysis.dmu) prints the metal '
                        'chemical potentials at its target_temperatures in this form.')
                mu[m] = float(value)
            tdmu[T] = {f'{a}2{b}': mu[b] - mu[a]
                       for i, a in enumerate(metals) for b in metals[i + 1:]}
        return tdmu

    def get_internal_map(self, temperatures):
        spec = self.section('adsorbates').get('internal_partition_functions')
        if spec is None:
            return None
        if not isinstance(spec, dict):
            raise ConfigError("'adsorbates.internal_partition_functions' must map "
                              'every fragment to its settings.')
        return internal_partition_functions(spec, temperatures)

    def get_volume(self, template):
        """Accessible volume: cell area times the gas height minus the
        covalent spheres of the metal atoms in the template."""
        spec = self.section('adsorbates')
        if spec.get('volume') is not None:
            return float(spec['volume'])
        if spec.get('gas_height') is None:
            raise ConfigError("Give 'adsorbates.gas_height' or 'adsorbates.volume'.")
        volume = np.linalg.norm(np.cross(template.cell[0], template.cell[1]))
        volume *= float(spec['gas_height'])
        for element, number in Counter(template.symbols).items():
            if element in self.metal_pool and element != 'X':
                radius = ase_data.covalent_radii[ase_data.atomic_numbers[element]]
                volume -= number * 4 / 3 * math.pi * radius**3
        return volume

    def get_adsorption_sites(self, template):
        from acat.adsorption_sites import SlabAdsorptionSites
        kwargs = {'surface': 'autoadsorbate', 'composition_effect': False,
                  'ignore_sites': self.section('emc').get('ignore_sites')}
        kwargs.update(self.section('adsorbates').get('adsorption_sites') or {})
        return SlabAdsorptionSites(template, **kwargs)

    def get_adsorbate_species(self):
        spec = self.section('adsorbates')
        _check_keys(spec, ('adsorbate_species', 'other_adsorbate_species',
                           'internal_partition_functions', 'gas_height', 'volume',
                           'adsorption_sites'), 'adsorbates')
        species = _as_list(spec.get('adsorbate_species')) or None
        other = spec.get('other_adsorbate_species')
        return species, (None if other is None else _as_list(other))


# ---------------------------------------------------------------------------
# catemc generate
# ---------------------------------------------------------------------------

def generate(cfg, force=False, log=print):
    """Generate the initial population (``catemc generate``).

    Decorates the template structure (``population.template``), writes
    ``{name}_input_pop.traj`` to the calculation directory and returns the
    population.
    """
    from catemc.generate import generate_population, assign_mu_windows

    pop_path = cfg.output('_input_pop.traj')
    if cfg.section('population').get('input_pop') is not None:
        raise ConfigError(
            "'population.input_pop' points to an existing population, so there "
            'is nothing to generate. Remove that entry to build a new population.')
    if os.path.isfile(pop_path) and not force:
        raise ConfigError(f'{pop_path} already exists. Use --force to overwrite it.')
    os.makedirs(cfg.directory, exist_ok=True)

    spec = dict(cfg.section('population'))
    own_keys = ('template', 'temperatures', 'replicas_per_temperature', 'seed',
                'variants', 'target_metal_composition',
                'target_nonmetal_composition', 'round_decimals', 'input_pop')
    reserved = ('template', 'pop_size', 'target_metal_species',
                'target_metal_composition', 'target_nonmetal_species',
                'target_nonmetal_composition', 'allowed_indices',
                'adsorption_sites', 'adsorption_dict', 'trajectory',
                'append_trajectory', 'seed')
    passthrough = [p for p in inspect.signature(generate_population).parameters
                   if p not in reserved]
    _check_keys(spec, own_keys + tuple(passthrough), 'population')
    temperatures = cfg.temperatures
    n_temp = len(temperatures)
    pop_size = cfg.pop_size
    seed = spec.get('seed')
    if seed is None:
        seed = random.randrange(2**31)
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    log(f'Random seed: {seed}')

    # 1) Template structure
    template = cfg.get_template()
    log(f'Template: {template.get_chemical_formula()} ({len(template)} sites)')
    regions = cfg.get_regions(template)
    allowed = regions['free_indices'] if regions['fix_indices'] else None

    # 2) Random decorations, one batch per variant
    shared = {k: spec[k] for k in passthrough if k in spec}
    if cfg.is_gc:
        shared.setdefault('adsorbate_species', cfg.get_adsorbate_species()[0])
        shared.setdefault('adsorbate_adsorbate_dmin',
                          cfg.section('emc').get('adsorbate_adsorbate_dmin', 2.))
    variants = [dict(v or {}) for v in _as_list(spec.get('variants'))] or [{}]
    sizes = [v.get('size') for v in variants]
    n_free = sum(s is None for s in sizes)
    rest = pop_size - sum(int(s) for s in sizes if s is not None)
    if rest < 0 or (n_free == 0 and rest != 0):
        raise ConfigError('The variant sizes must add up to the population size '
                          f'({pop_size}).')
    share = iter([rest // n_free + (i < rest % n_free) for i in range(n_free)]
                 if n_free else [])
    sizes = [int(s) if s is not None else next(share) for s in sizes]

    def variant_composition(variant, key, default):
        value = variant.get(key, spec.get(key, default))
        if value is None or value == 'template':
            return None
        return parse_formula(value) if isinstance(value, str) else {
            str(k): float(v) for k, v in value.items()}

    images = []
    sites = None
    cwd = os.getcwd()
    os.chdir(cfg.directory)   # generate_population writes scratch files here
    try:
        for v, (variant, size) in enumerate(zip(variants, sizes)):
            _check_keys(variant, ('size', 'target_metal_composition',
                                  'target_nonmetal_composition') + tuple(passthrough),
                        f'population.variants[{v}]')
            if size == 0:
                continue
            metals = variant_composition(variant, 'target_metal_composition', cfg.metals)
            if metals is not None and set(metals) != set(cfg.metal_pool):
                raise ConfigError(
                    f'population.variants[{v}].target_metal_composition must contain '
                    f"the same elements as the composition ({', '.join(cfg.metal_pool)}).")
            # Surface slabs keep the non-metal lattice of the template unless
            # a composition is requested explicitly.
            nonmetals = variant_composition(
                variant, 'target_nonmetal_composition', None if cfg.is_gc else cfg.nonmetals)
            extra = dict(shared)
            extra.update({k: variant[k] for k in passthrough if k in variant})
            if not extra.get('adsorbate_species'):
                # No adsorbates in this variant (bare bulk or clean surface).
                for key in ('adsorbate_species', 'adsorbate_coverage',
                            'adsorbate_adsorbate_dmin', 'max_adsorbate_species'):
                    extra.pop(key, None)
            else:
                extra['adsorbate_species'] = _as_list(extra['adsorbate_species'])
                if sites is None:
                    sites = cfg.get_adsorption_sites(template)
                extra['adsorption_sites'] = sites
            batch = generate_population(
                template=template.copy(),
                pop_size=size,
                target_metal_species=cfg.metal_pool if metals else None,
                target_metal_composition=metals,
                target_nonmetal_species=list(nonmetals) if nonmetals else None,
                target_nonmetal_composition=nonmetals,
                allowed_indices=allowed,
                seed=seed + v,
                **extra)
            images += [(v, atoms.copy()) for atoms in batch]
    finally:
        os.chdir(cwd)
    if len(images) != pop_size:
        raise ConfigError(f'Only {len(images)} of the {pop_size} requested '
                          'structures could be generated.')

    # 4) Temperatures, replica numbers and mu-windows
    if cfg.is_gc:
        mu_elements = list(cfg.get_mu_elements())

        def content(item):
            return sum(1 for s in item[1].get_chemical_symbols() if s in mu_elements)
        images = [item for v in range(len(variants)) for item in
                  sorted((it for it in images if it[0] == v), key=content)]
    blocks = [images[t::n_temp] for t in range(n_temp)]
    population = []
    for T, block in zip(temperatures, blocks):
        if cfg.is_gc:
            block = sorted(block, key=content)
        for v, atoms in block:
            atoms.info = {'key_value_pairs': {}, 'data': {}}
            atoms.info['key_value_pairs']['temperature'] = T
            atoms.info['key_value_pairs']['replica'] = len(population)
            atoms.info['key_value_pairs']['emc_step'] = 0
            if len(variants) > 1:
                atoms.info['key_value_pairs']['variant'] = v
            population.append(atoms)

    if cfg.is_gc:
        column, _, _ = cfg.get_element_maps()
        population = assign_mu_windows(
            population, reaction_conditions=make_grid(cfg.get_mu_axes()),
            element_column_map=column,
            round_decimals=int(spec.get('round_decimals', 3)))

    for atoms in population:
        kvp = atoms.info['key_value_pairs']
        line = f"Replica {kvp['replica']:3d}  T = {kvp['temperature']} K  "
        line += atoms.get_chemical_formula()
        if 'mu_window' in atoms.info['data']:
            mu = tuple(float(m) for m in atoms.info['data']['mu'])
            line += f"  {len(atoms.info['data']['mu_window'])} mu-points, now at {mu}"
        log(line)
    write(pop_path, population)
    log(f'Wrote {len(population)} replicas to {pop_path}')
    return population


# ---------------------------------------------------------------------------
# catemc run
# ---------------------------------------------------------------------------

def get_emc_class(cfg):
    if cfg.ensemble == 'canonical':
        from catemc.emc import EMC
        return EMC
    if cfg.ensemble == 'gibbs':
        from catemc.geemc import GEEMC
        return GEEMC
    from catemc.gcemc import GCEMC
    return GCEMC


def get_emc_kwargs(cfg, with_objects=True):
    """Constructor arguments of the EMC class selected by ``ensemble``.

    With ``with_objects=False`` the input population and the adsorption
    sites are not created, which is sufficient for ``catemc check``.
    """
    cls = get_emc_class(cfg)
    parameters = inspect.signature(cls.__init__).parameters
    user = dict(cfg.section('emc'))
    _check_keys(user, [p for p in parameters if p not in OBJECT_KEYS | {'self'}], 'emc')

    if cfg.is_gc:
        from catemc.calculation_templates import calculate_surface_fitness as fitness
    else:
        from catemc.calculation_templates import calculate_bulk_fitness as fitness
    factory_kwargs, fmax_list = cfg.get_calculator_factory_kwargs()

    template = cfg.get_template()
    regions = cfg.get_regions(template)
    kwargs = {
        'name': cfg.name,
        'calculation_dir': cfg.directory,
        'calculator_factory': create_calculators,
        'calculator_factory_kwargs': factory_kwargs,
        'fitness_kwargs': cfg.get_fitness_kwargs(fitness, fmax_list),
        'fix_indices': regions['fix_indices'],
        'metal_element_pool': cfg.metal_pool,
        'nonmetal_element_pool': cfg.nonmetal_pool,
        'mutable_metal_indices': regions['mutable_metal_indices'],
        'mutable_nonmetal_indices': regions['mutable_nonmetal_indices'],
    }
    if cfg.is_gc:
        species, other_species = cfg.get_adsorbate_species()
        column, eref, y2x = cfg.get_element_maps()
        kwargs.update({
            'bulk_metal_composition': {k: v for k, v in cfg.metals.items() if k != 'X'},
            'adsorbate_species': species,
            'other_adsorbate_species': other_species,
            'slab_indices': regions['slab_indices'],
            'volume': cfg.get_volume(template),
            'reaction_conditions': make_grid(cfg.get_mu_axes(), cfg.temperatures),
            'element_column_map': column,
            'element_eref_map': eref,
            'element_y2x_map': y2x,
            'temperature_species_internal_map': cfg.get_internal_map(cfg.temperatures),
        })
        try:
            kwargs['temperature_dmu_map'] = cfg.get_metal_dmu_map(cfg.temperatures)
        except ConfigError as err:
            if with_objects:
                raise
            kwargs['temperature_dmu_map'] = f'NOT AVAILABLE YET: {err}'
    if user.get('pressure') is not None:
        user['pressure'] = float(user['pressure']) * units.bar
    if user.get('reaction_conditions') is not None:
        user['reaction_conditions'] = np.array(user['reaction_conditions'], dtype=float)
    kwargs.update(user)

    if with_objects:
        if not os.path.isfile(cfg.population_path):
            raise ConfigError(f'{cfg.population_path} not found. Run '
                              '"catemc generate" first.')
        kwargs['input_pop'] = read(cfg.population_path, index=':')
        if cfg.is_gc:
            kwargs['adsorption_sites'] = cfg.get_adsorption_sites(template)
    return kwargs


def run(cfg, log=print):
    """Run the simulation (``catemc run``)."""
    spec = cfg.section('run')
    _check_keys(spec, ('total_steps', 'starting_step', 'restart_with_trim'), 'run')
    cls = get_emc_class(cfg)
    kwargs = get_emc_kwargs(cfg)
    log(f"{cls.__name__} '{cfg.name}': {len(kwargs['input_pop'])} replicas in "
        f'{cfg.directory}')
    emc = cls(**kwargs)
    emc.run(**spec)
    return emc


# ---------------------------------------------------------------------------
# catemc analyze
# ---------------------------------------------------------------------------

def _task(spec, key):
    """Settings of one analysis task, or ``None`` when it is not requested."""
    value = spec.get(key)
    if value is None or value is False:
        return None
    return {} if value is True else value


def _method_kwargs(method, common, task, where, skip=(), extra=()):
    """Arguments of ``method`` taken from the common and task-level entries."""
    names = [p for p in inspect.signature(method).parameters
             if p not in ('self',) + tuple(skip)]
    _check_keys(task, tuple(names) + tuple(extra), where)
    kwargs = {k: common[k] for k in names if k in common}
    kwargs.update({k: v for k, v in task.items() if k in names})
    return kwargs


def get_analysis_conditions(cfg, conditions):
    """Reaction conditions of a grand-canonical analysis task.

    The simulated states (the grid of ``chemical_potentials.nonmetals`` at
    every simulated temperature) always come first, because MBAR needs
    them. The entries of ``reaction_conditions`` follow. In every entry each
    reservoir element is either a number, the absolute chemical potential
    in eV, or a mapping ``{reference, range, step}`` (or ``{reference,
    values}``) for the chemical potentials ``reference + values``. ``T`` is
    one temperature or a list of them.

    Returns the condition array (chemical potentials relative to the
    references of ``chemical_potentials.nonmetals``, temperature in the last
    column), the row indices of the ``reaction_conditions`` and the row
    indices of the entries that are one single condition.
    """
    column, eref, _ = cfg.get_element_maps()
    grid = make_grid(cfg.get_mu_axes(), cfg.temperatures)
    blocks, points = [grid], []
    n = len(grid)
    for i, cond in enumerate(_as_list(conditions)):
        where = f'analysis.reaction_conditions[{i}]'
        if not isinstance(cond, dict):
            raise ConfigError(f"{where} must be a mapping with one entry per "
                              "reservoir element and the temperature 'T'.")
        _check_keys(cond, list(column) + ['T'], where)
        if cond.get('T') is None:
            raise ConfigError(f"{where} needs the temperature 'T' (one value or a list).")
        axes = []
        for element in column:
            if element not in cond:
                raise ConfigError(f"{where} misses the element '{element}'.")
            value = cond[element]
            if isinstance(value, dict):
                par = _mu_spec(value, f'{where}.{element}')
                axis = (float(par['reference']) - eref[element]
                        + _axis(par, f'{where}.{element}'))
            else:
                axis = np.array([float(value) - eref[element]])
            axes.append(np.round(axis, 10))
        block = make_grid(axes, [float(T) for T in _as_list(cond['T'])])
        if len(block) == 1:
            points.append(n)
        blocks.append(block)
        n += len(block)
    return np.vstack(blocks), list(range(len(grid), n)), points


def format_metal_potentials(dmu, metals):
    """Lines of a ``chemical_potentials.metals`` block from the output of
    :meth:`catemc.analysis.GEEMCAnalysis.get_dmu_estimates`: the chemical
    potential of every metal at every temperature, relative to the first
    metal."""
    metals = [m for m in metals if m != 'X']
    lines = ['  metals:']
    for m in metals:
        values = []
        for T, pairs in dmu.items():
            if m == metals[0]:
                mu = 0.
            elif f'{metals[0]}2{m}' in pairs:
                mu = float(pairs[f'{metals[0]}2{m}'])
            else:
                mu = -float(pairs[f'{m}2{metals[0]}'])
            T = int(T) if float(T).is_integer() else float(T)
            values.append(f'{T}: {mu:.4f}')
        lines.append(f"    {m}: {{{', '.join(values)}}}")
    return lines


def analyze(cfg, fresh=False, log=print):
    """Post-simulation analysis (``catemc analyze``).

    Only the tasks listed in the ``analysis`` section are executed. The
    analysis object is returned so that it can be used interactively.
    """
    from catemc import analysis as ana_module

    spec = dict(cfg.section('analysis'))
    emc = cfg.section('emc')
    if cfg.is_gc:
        cls = ana_module.GCEMCAnalysis
        tasks = ('trajectory', 'diagnostics', 'expectations', 'fes')
        methods = (cls.prepare_data, cls.get_diagnostics,
                   cls.get_structural_expectations, cls.get_structural_fes)
    elif cfg.ensemble == 'canonical':
        cls = ana_module.EMCAnalysis
        tasks = ('trajectory', 'diagnostics', 'sro')
        methods = (cls.prepare_data, cls.get_diagnostics, cls.get_sro)
    else:
        cls = ana_module.GEEMCAnalysis
        tasks = ('trajectory', 'diagnostics', 'sro', 'dmu')
        methods = (cls.prepare_data, cls.get_diagnostics, cls.get_sro,
                   cls.get_dmu_estimates)
    init_names = [p for p in inspect.signature(cls.__init__).parameters
                  if p not in ('self', 'name', 'calculation_dir')]
    common_names = {p for m in methods for p in inspect.signature(m).parameters
                    if p != 'self'}
    _check_keys(spec, set(tasks) | set(init_names) | common_names |
                ({'reaction_conditions'} if cfg.is_gc else set()), 'analysis')
    common = {k: v for k, v in spec.items() if k not in tasks}

    init = {'pop_size': cfg.pop_size,
            'target_metal_species': [m for m in cfg.metal_pool if m != 'X']}
    if cfg.is_gc:
        species, _ = cfg.get_adsorbate_species()
        init['target_metal_composition'] = {
            k: v for k, v in cfg.metals.items() if k != 'X'}
        init['adsorbate_species'] = species
    else:
        if emc.get('pressure') is not None:
            init['pressure'] = float(emc['pressure']) * units.bar
        key = 'category_key' if cfg.ensemble == 'canonical' else 'interswap_within_key'
        if emc.get(key) is not None:
            init[key] = emc[key]
    for key in ('trim_generation', 'temperature_key', 'replica_key', 'acceptance_key'):
        if key in emc:
            init[key] = emc[key]
    init.update({k: spec[k] for k in init_names if k in spec})
    ana = cls(name=cfg.name, calculation_dir=cfg.directory, **init)

    if fresh:
        cached = ['_mbar_data.pkl', '_mbar_mc_data.pkl', '_mbar_widom_data.pkl',
                  '_mbar.traj']
        for suffix in cached:
            if os.path.isfile(cfg.output(suffix)):
                os.remove(cfg.output(suffix))
    prepare = {'raw_score_cutoff': emc.get('raw_score_cutoff', -10000.)}
    prepare.update({k: common[k] for k in ('use_exist', 'raw_score_cutoff')
                    if k in common})
    if fresh:
        prepare['use_exist'] = False
    ana.prepare_data(**prepare)
    results = {}

    task = _task(spec, 'trajectory')
    if task is not None:
        if isinstance(task, dict):
            _check_keys(task, ('replicas',), 'analysis.trajectory')
            replicas = task.get('replicas', range(cfg.pop_size))
        else:
            replicas = task
        for replica in (replicas if isinstance(replicas, range) else _as_list(replicas)):
            ana.get_trajectory(replica=int(replica))
            log(f'Wrote the trajectory of replica {replica}.')

    task = _task(spec, 'diagnostics')
    if task is not None:
        results['diagnostics'] = ana.get_diagnostics(**_method_kwargs(
            cls.get_diagnostics, common, task, 'analysis.diagnostics'))
        log(f"Wrote {cfg.output('_mbar_diagnostics.pkl')}")

    if not cfg.is_gc:
        task = _task(spec, 'sro')
        if task is not None:
            results['sro'] = ana.get_sro(**_method_kwargs(
                cls.get_sro, common, task, 'analysis.sro'))
            log('Wrote the Warren-Cowley short-range order parameters.')
        task = _task(spec, 'dmu')
        if task is not None:
            kwargs = _method_kwargs(cls.get_dmu_estimates, common, task, 'analysis.dmu')
            dmu = ana.get_dmu_estimates(**kwargs)
            results['dmu'] = dmu
            if emc.get('interswap_within_key') is None:
                log('Metal chemical-potential differences mu_B - mu_A (eV), '
                    "listed as 'A2B':")
                for T, values in dmu.items():
                    log(f'  {T} K: ' + ', '.join(
                        f'{k} {v:.3f}' for k, v in values.items()))
                if kwargs.get('wls_correction', True) and not kwargs.get('excess_only'):
                    log('Metal chemical potentials for chemical_potentials.metals '
                        'of a surface run (eV):')
                    for line in format_metal_potentials(dmu, ana.target_metal_species):
                        log(line)
            log(f"Wrote {cfg.output('_mbar_dmu_estimates.pkl')}")
        return ana, results

    # Grand-canonical tasks
    template = cfg.get_template()
    column, eref, y2x = cfg.get_element_maps()
    shared = {
        'element_column_map': column,
        'element_eref_map': eref,
        'element_y2x_map': y2x,
        'slab_indices': cfg.get_regions(template)['slab_indices'],
    }
    for key, method in (('expectations', cls.get_structural_expectations),
                        ('fes', cls.get_structural_fes)):
        task = _task(spec, key)
        if task is None:
            continue
        task = dict(task)
        conditions, listed, extra = get_analysis_conditions(
            cfg, task.get('reaction_conditions', common.get('reaction_conditions')))
        kwargs = _method_kwargs(
            method, common, task, f'analysis.{key}', skip=('reaction_conditions',),
            extra=('reaction_conditions',))
        kwargs.update(shared)
        target_temperatures = sorted(set(conditions[:, -1].tolist()))
        kwargs['temperature_dmu_map'] = cfg.get_metal_dmu_map(target_temperatures)
        if cfg.section('adsorbates').get('internal_partition_functions') is not None:
            kwargs['temperature_species_internal_map'] = cfg.get_internal_map(
                target_temperatures)
        if key == 'fes' and listed and 'sample_condition_indices' not in kwargs:
            kwargs['sample_condition_indices'] = listed
        results[key] = method(ana, reaction_conditions=conditions, **kwargs)
        np.save(cfg.output(f'_mbar_{key}_conditions.npy'), conditions)
        log(f'{key}: {len(conditions)} reaction conditions, saved to '
            f"{cfg.output(f'_mbar_{key}_conditions.npy')}")
        if key == 'expectations' and extra:
            values = results[key][0] if isinstance(results[key], tuple) else results[key]
            for i in extra:
                log(f'  condition {np.round(conditions[i], 6).tolist()}:')
                for cv, array in values.items():
                    if cv.startswith(('x_surf_', 'ads_coverage', 'x_lattice_vac')):
                        log(f'    {cv:24s} {array[i]:.4f}')
    return ana, results


# ---------------------------------------------------------------------------
# catemc check
# ---------------------------------------------------------------------------

def summarize(cfg, log=print):
    """Print what a configuration resolves to without running anything."""
    def row(label, value):
        log(f'{label:25s}: {value}')

    cls = get_emc_class(cfg)
    row('System', cfg.name)
    row('Calculation directory', cfg.directory)
    row('Ensemble', f'{cfg.ensemble} ({cls.__name__})')
    row('Metals', cfg.metals)
    row('Non-metals', cfg.nonmetals)
    row('Temperatures (K)', cfg.temperatures)
    row('Population size', cfg.pop_size)
    template = cfg.get_template()
    row('Template', f"{template.get_chemical_formula()} ({len(template)} sites; "
        f"{cfg.resolve(cfg.section('population')['template'])})")
    kwargs = get_emc_kwargs(cfg, with_objects=False)
    for key in ('fix_indices', 'mutable_metal_indices', 'mutable_nonmetal_indices',
                'slab_indices'):
        if key in kwargs:
            value = kwargs[key]
            row(key, ('none' if key == 'fix_indices' else 'all') if value is None
                else f'{len(value)} atoms')
    if cfg.is_gc:
        row('Accessible volume (A^3)', f"{kwargs['volume']:.2f}")
        row('Reaction conditions', f"{len(kwargs['reaction_conditions'])} states, "
            f"columns {list(kwargs['element_column_map'])} + T")
        row('Reference energies (eV)', kwargs['element_eref_map'])
        row('Bulk non-metal/metal y/x', kwargs['element_y2x_map'])
        row('Inserted fragments', kwargs['adsorbate_species'])
        row('Other fragments', kwargs['other_adsorbate_species'])
        internal = kwargs['temperature_species_internal_map']
        for T, values in (internal or {}).items():
            row(f'Q_int at {T} K', ', '.join(f'{k} {v:.4g}' for k, v in values.items()))
        tdmu = kwargs['temperature_dmu_map']
        row('Metal reservoir', 'none (no metal chemical potentials given)'
            if tdmu is None else tdmu if isinstance(tdmu, str)
            else f'chemical potentials at {sorted(tdmu)} K')
    row('Calculators', kwargs['calculator_factory_kwargs'])
    hidden = ('name', 'calculation_dir', 'calculator_factory',
              'calculator_factory_kwargs', 'fitness_kwargs', 'fix_indices',
              'mutable_metal_indices', 'mutable_nonmetal_indices', 'slab_indices',
              'volume', 'reaction_conditions', 'element_column_map',
              'element_eref_map', 'element_y2x_map', 'temperature_dmu_map',
              'temperature_species_internal_map', 'metal_element_pool',
              'nonmetal_element_pool', 'bulk_metal_composition',
              'adsorbate_species', 'other_adsorbate_species')
    row('Fitness (ASE units)', kwargs['fitness_kwargs'])
    row('EMC settings', {k: v for k, v in kwargs.items() if k not in hidden})
    row('Run settings', cfg.section('run'))
    row('Population file', cfg.population_path + (
        '' if os.path.isfile(cfg.population_path) else '  (not generated yet)'))
    return kwargs
