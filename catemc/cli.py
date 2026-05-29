"""CatEMC command-line interface.

Usage:
    catemc setup dopant_workflow              # emit template config YAML
    catemc setup dopant_workflow --input X    # scaffold from config
"""

import argparse
import sys


def cmd_setup(args):
    """Handle the 'setup' subcommand."""
    workflow_name = args.workflow

    if workflow_name == 'dopant_workflow':
        from catemc.workflows.dopant_workflow import emit_template, scaffold
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
        help='Workflow name (e.g. dopant_workflow)',
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

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == '__main__':
    main()
