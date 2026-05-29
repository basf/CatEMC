"""CatEMC command-line interface.

Usage:
    catemc setup ABOx_screening              # emit template config YAML
    catemc setup ABOx_screening --input X    # scaffold from config
    catemc overview                          # multi-system live dashboard
"""

import argparse
import sys


def cmd_setup(args):
    """Handle the 'setup' subcommand."""
    workflow_name = args.workflow

    if workflow_name == 'ABOx_screening':
        from catemc.workflows.ABOx_screening import emit_template, scaffold
        if args.input:
            scaffold(args.input, force=args.force)
        else:
            output = args.output or 'emc_config.yaml'
            emit_template(output, force=args.force)
    else:
        from catemc.workflows import AVAILABLE_WORKFLOWS
        print(f"Unknown workflow: {workflow_name}")
        print(f"Available workflows: {', '.join(AVAILABLE_WORKFLOWS)}")
        sys.exit(1)



def cmd_overview(args):
    """Handle the 'overview' subcommand."""
    from catemc.multi_overview import run
    run(
        base_dir=args.dir,
        refresh=args.refresh,
        depth=args.depth,
        sort_by=args.sort,
        once=args.once,
        stale_timeout=args.stale_timeout,
        tabular=getattr(args, "tabular", False),
    )


def main():
    parser = argparse.ArgumentParser(
        prog='catemc',
        description='CatEMC: Evolutionary Monte Carlo for heterogeneous catalysis',
    )
    subparsers = parser.add_subparsers(dest='command')

    # --- setup subcommand ---
    setup_parser = subparsers.add_parser(
        'setup',
        help='Scaffold a workflow directory from a config template',
    )
    setup_parser.add_argument(
        'workflow',
        help='Workflow name (e.g. ABOx_screening)',
    )
    setup_parser.add_argument(
        '--input', '-i',
        metavar='CONFIG',
        help='Path to filled-in config YAML. If omitted, emits a template.',
    )
    setup_parser.add_argument(
        '--output', '-o',
        metavar='FILE',
        help='Output path for template YAML (default: emc_config.yaml)',
    )
    setup_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Overwrite existing files without prompting',
    )
    setup_parser.set_defaults(func=cmd_setup)

    # --- overview subcommand ---
    overview_parser = subparsers.add_parser(
        'overview',
        help='Live multi-system overview dashboard',
    )
    overview_parser.add_argument(
        '--dir', '-d', default='.',
        help='Base directory to scan (default: cwd)',
    )
    overview_parser.add_argument(
        '--refresh', '-r', type=float, default=5.0,
        help='Refresh interval in seconds (default: 5.0)',
    )
    overview_parser.add_argument(
        '--depth', type=int, default=2,
        help='Max subdirectory depth to search (default: 2)',
    )
    overview_parser.add_argument(
        '--sort', '-s', default='status',
        choices=['status', 'name', 'gen', 'energy', 'acceptance'],
        help='Sort order (default: status)',
    )
    overview_parser.add_argument(
        '--once', action='store_true',
        help='Print snapshot once and exit (no live updates)',
    )
    overview_parser.add_argument(
        '--stale-timeout', type=int, default=300,
        help='Seconds before a run is marked STALLED (default: 300)',
    )
    overview_parser.add_argument(
        '--tabular', action='store_true',
        help='Use simple table layout instead of cards dashboard',
    )
    overview_parser.set_defaults(func=cmd_overview)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == '__main__':
    main()
