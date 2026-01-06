"""Command-line interface for DBSIpy.

This module provides the CLI entry point for running DBSI (Diffusion Basis Spectrum Imaging)
analysis on diffusion MRI data. It handles argument parsing, configuration file validation,
and orchestrates the analysis pipeline.

The CLI supports multiple tissue types and analysis engines:
- DBSI: Classic diffusion basis spectrum imaging
- IA: Intra-axonal/Extra-axonal separation

Example:
    DBSI run --cfg_path path/to/config.ini
    DBSI run  # Launches GUI for file selection
"""

import argparse
from dbsipy.core import runner

class CLI:
    def __init__(self, subparsers) -> None:
        """Initializes subparsers for input parameters

        :param subparsers: Parsers for each relevant module
        :type subparsers: str
        """   
        self.subparsers = subparsers        
        pass

    def validate_args(self, args):   
        """Validation step for parsed user input arguments

        :param args: Parsed user inputs
        :type args: dictionary
        :return: Parsed and validated arguments
        :rtype: dictionary
        """         
        """Validate parsed args.

        Note: master_cli.py passes a dict (via vars(...)). Other callers may pass an
        argparse.Namespace. Support both.
        """
        import os

        cfg_path = None
        if isinstance(args, dict):
            cfg_path = args.get('cfg_path', None)
        else:
            cfg_path = getattr(args, 'cfg_path', None)

        # Validate configuration file exists if provided
        if cfg_path is not None:
            cfg_path = str(cfg_path)
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(
                    f"Configuration file not found: {cfg_path}\n"
                    f"Please check the path and try again."
                )

            # Normalize back into args to keep downstream expectations consistent.
            if isinstance(args, dict):
                args['cfg_path'] = cfg_path
            else:
                setattr(args, 'cfg_path', cfg_path)

        return args
    
    def run(self, args):
        """Run computation using parsed user inputs

        :param args: User inputs for relevant parameters
        :type args: dictionary
        """
        runner.run(args)


    def add_subparser_args(self) -> argparse:
        """Defines subparsers for each computation parameter.

        :return: argparse object containing subparsers for each computation parameter
        :rtype: argparse
        """  
        
        subparser = self.subparsers.add_parser("run",
                                        description="run the computation",
                                        )

        """  Computation Parameters """

        subparser.add_argument("--cfg_path", nargs=None, type=str,
                            dest='cfg_path',
                            required=False,
                            help="The path to the configuration File")

        subparser.add_argument(
            "--output_mode",
            type=str,
            required=False,
            default=None,
            choices=["quiet", "standard", "verbose", "debug"],
            help="Terminal output mode (overrides config/UI): quiet | standard | verbose | debug",
        )
     
        return self.subparsers


