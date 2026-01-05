"""Package-scoped CLI entrypoint.

This is the preferred console entry target for installed DBSIpy.
"""

from __future__ import annotations

import sys
import argparse

from dbsipy.cli import CLI as runCLI
from dbsipy.benchmark_cli import BenchmarkCLI


def main() -> None:
    TOOL_DICT = {'run': runCLI, 'benchmark': BenchmarkCLI}

    parser = argparse.ArgumentParser(
        prog='DBSI',
        description='DBSIpy command line interface',
        epilog='See online documentation for more information about each function.',
    )
    subparsers = parser.add_subparsers(dest='command', help='sub-command help')

    # Register subcommands up front so help output is complete.
    commands = {}
    for name, factory in TOOL_DICT.items():
        cli = factory(subparsers)
        cli.add_subparser_args()
        commands[name] = cli

    argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        return

    # Let argparse handle -h/--help and unknown commands.
    ns = parser.parse_args(argv)
    command = getattr(ns, 'command', None)
    if not command:
        parser.print_help()
        return

    cli = commands.get(command)
    if cli is None:
        parser.print_help()
        parser.exit(2, f"\nUnknown command: {command!r}\n")

    args = cli.validate_args(vars(ns))
    cli.run(args)


if __name__ == "__main__":
    main()
