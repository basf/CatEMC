#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command-line interface of CatEMC.

One YAML file describes one system (composition, template structure,
population, calculators and sampling settings), and every step of a study
is one command acting on that file::

    catemc init gibbs -o config.yaml    # write a commented starting config
    catemc check config.yaml            # show what the config resolves to
    catemc generate config.yaml         # build the initial population
    catemc run config.yaml              # run (or restart) the simulation
    catemc analyze config.yaml          # post-simulation analysis

Any entry of the file can be replaced on the command line with
``-s section.key=value``, e.g. ``catemc run config.yaml -s run.total_steps=200``.

The CLI is a thin layer on top of the Python API. The translation from the
YAML file to the arguments of the CatEMC classes lives in
:mod:`catemc.config`.
"""
import argparse
import os
import sys
import warnings


def _add_config_arguments(parser):
    parser.add_argument('config', help='path of the YAML configuration file')
    parser.add_argument(
        '-s', '--set', dest='overrides', action='append', default=[],
        metavar='KEY=VALUE',
        help='replace one entry of the configuration file, e.g. '
             '-s run.total_steps=200 (can be given several times)')


def build_parser():
    from catemc import __version__

    parser = argparse.ArgumentParser(
        prog='catemc',
        description='Evolutionary Monte Carlo simulations of bulk materials '
                    'and catalyst surfaces from one YAML configuration file.',
        epilog='Documentation: https://catemc.readthedocs.io')
    parser.add_argument('--version', action='version',
                        version=f'catemc {__version__}')
    sub = parser.add_subparsers(dest='command', metavar='command')

    p = sub.add_parser(
        'init', help='write a commented configuration file to start from',
        description='Write a commented configuration file to start from. The '
                    'canonical and gibbs files run as they are with the EMT '
                    'calculator of ASE, so they double as a quick test of the '
                    'installation.')
    p.add_argument('ensemble', nargs='?', default='gibbs',
                   choices=('canonical', 'gibbs', 'grand_canonical'),
                   help='statistical ensemble of the simulation (default: gibbs)')
    p.add_argument('-o', '--output', default='config.yaml',
                   help='file to write (default: config.yaml)')
    p.add_argument('-f', '--force', action='store_true',
                   help='overwrite an existing file')

    p = sub.add_parser(
        'check', help='show what a configuration resolves to',
        description='Validate a configuration file and print the settings '
                    'CatEMC derives from it, without running anything.')
    _add_config_arguments(p)

    p = sub.add_parser(
        'generate', help='generate the initial population',
        description='Decorate the template structure (population.template) '
                    'and write the initial population ({name}_input_pop.traj).')
    _add_config_arguments(p)
    p.add_argument('-f', '--force', action='store_true',
                   help='overwrite an existing initial population')

    p = sub.add_parser(
        'run', help='run or restart the simulation',
        description='Run the simulation. Running the command again continues '
                    'from the last completed step.')
    _add_config_arguments(p)

    p = sub.add_parser(
        'analyze', help='run the post-simulation analysis',
        description="Run the tasks listed in the 'analysis' section of the "
                    'configuration file.')
    _add_config_arguments(p)
    p.add_argument('--fresh', action='store_true',
                   help='recollect the samples from the databases instead of '
                        'reusing the cached analysis data')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    # ACAT imports the deprecated ase.ga module, which prints a long notice.
    warnings.filterwarnings('ignore', category=FutureWarning, module='ase.ga')
    # Imported here so that "catemc --help" stays fast.
    from catemc import config as cfg_module

    try:
        if args.command == 'init':
            from catemc.config_templates import TEMPLATES
            if os.path.exists(args.output) and not args.force:
                raise cfg_module.ConfigError(
                    f'{args.output} already exists. Use --force to overwrite it.')
            with open(args.output, 'w') as handle:
                handle.write(TEMPLATES[args.ensemble])
            print(f'Wrote {args.output}. Provide the template structure named '
                  'in population.template, edit the file, then run:\n'
                  f'  catemc generate {args.output}\n'
                  f'  catemc run {args.output}\n'
                  f'  catemc analyze {args.output}')
            return 0

        if not os.path.isfile(args.config):
            raise cfg_module.ConfigError(f'{args.config} not found.')
        cfg = cfg_module.Config(args.config, overrides=args.overrides)
        if args.command == 'check':
            cfg_module.summarize(cfg)
        elif args.command == 'generate':
            cfg_module.generate(cfg, force=args.force)
        elif args.command == 'run':
            cfg_module.run(cfg)
        elif args.command == 'analyze':
            cfg_module.analyze(cfg, fresh=args.fresh)
    except cfg_module.ConfigError as err:
        print(f'catemc: error: {err}', file=sys.stderr)
        return 2
    except BrokenPipeError:
        # The output was piped into a command that stopped reading (e.g. head).
        sys.stderr.close()
        return 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
