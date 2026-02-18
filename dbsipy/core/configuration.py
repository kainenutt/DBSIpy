import configparser
import csv
import logging
import os

import numpy as np
import torch

from dbsipy.core.validation import ConfigurationError
from dbsipy.configs.paths import resolve_basis_path


class configuration:
    def __init__(self, cfg_file) -> None:
        self.cfg_file = cfg_file
        # Convert basis-set paths before validation so cross-OS configs (e.g., Z:\ paths)
        # don't fail early file-existence checks.
        self._convert_basis_set_paths(cfg_file)
        self._validate_config(cfg_file)  # Validate before setup
        self._setup_config(cfg_file)
        pass

    @staticmethod
    def _normalize_output_mode(value: str | None) -> str:
        if value is None:
            return 'standard'
        v = str(value).strip().lower()
        if v in {'quiet', 'q'}:
            return 'quiet'
        if v in {'standard', 'std', 'default'}:
            return 'standard'
        if v in {'verbose', 'v'}:
            return 'verbose'
        if v in {'debug', 'dbg'}:
            return 'debug'
        raise ConfigurationError(
            "Invalid output_mode value.\n"
            "Valid options: quiet | standard | verbose | debug\n"
            f"Current value: '{value}'"
        )

    @property
    def output_mode(self) -> str:
        return str(getattr(self, '_output_mode', 'standard'))

    @property
    def diagnostics_enabled(self) -> bool:
        # Developer/debug stream should only be enabled when explicitly requested.
        return self.output_mode == 'debug'

    @staticmethod
    def _normalize_output_map_set(value: str | None) -> str:
        if value is None:
            # Preserve historical behavior: allocate default + expanded maps.
            return 'expanded'
        value_norm = str(value).strip().lower()
        # Legacy synonyms
        if value_norm in {'both', 'all'}:
            return 'expanded'
        if value_norm in {'default', 'defaults'}:
            return 'default'
        if value_norm in {'expanded', 'exp'}:
            return 'expanded'
        raise ConfigurationError(
            "Invalid output_map_set value.\n"
            "Valid options: default | expanded\n"
            f"Current value: '{value}'"
        )

    @staticmethod
    def _normalize_signal_normalization(value: str | None) -> str:
        if value is None:
            return 'auto'
        value_norm = str(value).strip().lower()
        if value_norm in {'auto'}:
            return 'auto'
        if value_norm in {'max'}:
            return 'max'
        if value_norm in {'b0', 'b_0'}:
            return 'b0'
        if value_norm in {'minb', 'min_b'}:
            return 'minb'
        if value_norm in {'none', 'raw'}:
            return 'none'
        raise ConfigurationError(
            "Invalid signal_normalization value.\n"
            "Valid options: auto | max | b0 | minb | none\n"
            f"Current value: '{value}'"
        )

    def _convert_basis_set_paths(self, input_cfg_file) -> None:
        """Convert basis set paths between Windows and Linux.

        Converts Windows (Z:\\) and Linux (/bmrc-homes/nmrgrp/nmr202/) roots
        based on current OS and file existence.
        """
        # Prefer portable resolution into the installed package's BasisSets.
        # This lets shipped configs work unchanged across OS and install locations.
        cfg_source = None
        try:
            if input_cfg_file.has_section('DEBUG') and input_cfg_file.has_option('DEBUG', 'cfg_source'):
                cfg_source = str(input_cfg_file.get('DEBUG', 'cfg_source')).strip() or None
        except Exception:
            cfg_source = None

        basis_sections = ['STEP_1', 'STEP_2']
        basis_keys = {'angle_basis', 'iso_basis', 'step_2_axials', 'step_2_radials'}

        for section in basis_sections:
            if not input_cfg_file.has_section(section):
                continue

            for key in list(input_cfg_file[section].keys()):
                if key not in basis_keys:
                    continue

                raw = str(input_cfg_file.get(section, key, fallback='')).strip()
                if not raw:
                    continue

                resolved = resolve_basis_path(raw, cfg_source=cfg_source)
                if resolved and resolved != raw:
                    input_cfg_file.set(section, key, resolved)
                    logging.debug(f"Resolved basis path {section}.{key}: {raw} -> {resolved}")

    def _validate_config(self, input_cfg_file) -> None:
        """Validate configuration file for required sections/options and file paths."""
        required_sections = ['INPUT', 'GLOBAL']
        for section in required_sections:
            if not input_cfg_file.has_section(section):
                raise ConfigurationError(
                    f"Missing required section [{section}] in configuration file.\n"
                    f"Check your .ini file and ensure all required sections are present."
                )

        required_inputs = {
            'dwi_file': 'DWI/DTI data file (NIfTI format)',
            'bval_file': 'B-values file (text)',
            'bvec_file': 'B-vectors file (text)',
        }

        for key, description in required_inputs.items():
            if not input_cfg_file.has_option('INPUT', key):
                raise ConfigurationError(
                    f"Missing required field '{key}' in [INPUT] section.\n"
                    f"This should specify the {description}.\n"
                    f"Add '{key} = /path/to/file' to your configuration."
                )

            file_path = input_cfg_file['INPUT'][key]
            if not os.path.exists(file_path):
                raise ConfigurationError(
                    f"File not found: {file_path}\n"
                    f"Specified in configuration as '{key}'.\n"
                    f"Check that the path is correct and the file exists."
                )

        if input_cfg_file.has_option('INPUT', 'mask_file'):
            mask_path = str(input_cfg_file['INPUT']['mask_file']).strip()
            none_like = {'n/a', 'n\\a', 'na', 'none'}
            if mask_path and mask_path.lower() not in ({'auto'} | none_like):
                if not os.path.exists(mask_path):
                    raise ConfigurationError(
                        f"Mask file not found: {mask_path}\n"
                        f"Specified in configuration as 'mask_file'.\n"
                        f"Use 'mask_file = auto' to auto-generate a mask, leave it blank for a minimal signal mask, "
                        f"or provide a valid path."
                    )

        if not input_cfg_file.has_option('GLOBAL', 'model_engine'):
            raise ConfigurationError(
                "Missing required field 'model_engine' in [GLOBAL] section.\n"
                "Valid options: DBSI, IA, DTI, NODDI\n"
                "Add 'model_engine = DBSI' (or IA/DTI/NODDI) to your configuration."
            )

        engine = input_cfg_file['GLOBAL']['model_engine']
        valid_engines = ['DBSI', 'IA', 'DTI', 'NODDI']
        if engine not in valid_engines:
            raise ConfigurationError(
                f"Invalid model_engine: '{engine}'\n"
                f"Valid options are: {', '.join(valid_engines)}\n"
                f"Update your configuration to use one of these engines."
            )

        if engine in ['DBSI', 'IA']:
            if not input_cfg_file.has_section('STEP_1'):
                raise ConfigurationError(
                    f"Engine '{engine}' requires [STEP_1] section with basis set paths.\n"
                    f"Add [STEP_1] section with 'angle_basis' and 'iso_basis' paths."
                )
            if not input_cfg_file.has_section('STEP_2'):
                raise ConfigurationError(
                    f"Engine '{engine}' requires [STEP_2] section with basis set paths.\n"
                    f"Add [STEP_2] section with 'axial_basis' and 'radial_basis' paths."
                )

            required_by_section: dict[str, dict[str, str]] = {
                'GLOBAL': {
                    'max_group_number': 'Maximum number of fibers to model',
                    'fiber_threshold': 'Anisotropic signal threshold',
                },
                'STEP_1': {
                    'angle_threshold': 'Angular threshold for anisotropic grouping (deg)',
                    'angle_basis': 'Angle basis CSV file path',
                    'iso_basis': 'Isotropic basis CSV file path',
                    'step_1_axial': 'Step 1 axial diffusivity (mm^2/s)',
                    'step_1_radial': 'Step 1 radial diffusivity (mm^2/s)',
                },
                'STEP_2': {
                    'step_2_axials': 'Step 2 axial basis CSV file path',
                    'step_2_radials': 'Step 2 radial basis CSV file path',
                    'intra_threshold': 'IA intra-axonal RD threshold (mm^2/s)',
                },
                'ISOTROPIC': {
                    'restricted_threshold': 'Restricted isotropic threshold (mm^2/s)',
                    'free_water_threshold': 'Free-water isotropic threshold (mm^2/s)',
                },
            }

            missing: list[str] = []
            for section, opts in required_by_section.items():
                if not input_cfg_file.has_section(section):
                    missing.append(f"Missing required section [{section}]")
                    continue
                for opt, desc in opts.items():
                    if not input_cfg_file.has_option(section, opt):
                        missing.append(f"Missing [{section}] {opt} ({desc})")
                        continue
                    val = str(input_cfg_file.get(section, opt, fallback='')).strip()
                    if val == '':
                        missing.append(f"Empty [{section}] {opt} ({desc})")

            if missing:
                raise ConfigurationError(
                    "Configuration missing required DBSI/IA options.\n"
                    "This often happens when selecting a base config (.ini) for a different engine.\n\n"
                    + "\n".join(f"- {m}" for m in missing)
                )

            basis_paths = [
                ('STEP_1', 'angle_basis'),
                ('STEP_1', 'iso_basis'),
                ('STEP_2', 'step_2_axials'),
                ('STEP_2', 'step_2_radials'),
            ]
            missing_files: list[str] = []

            cfg_source = None
            try:
                if input_cfg_file.has_section('DEBUG') and input_cfg_file.has_option('DEBUG', 'cfg_source'):
                    cfg_source = str(input_cfg_file.get('DEBUG', 'cfg_source', fallback=None) or '').strip() or None
            except Exception:
                cfg_source = None

            for sec, opt in basis_paths:
                try:
                    p = str(input_cfg_file.get(sec, opt)).strip()
                except Exception:
                    p = ''
                if not p:
                    missing_files.append(f"[{sec}] {opt} -> <empty>")
                    continue

                try:
                    from dbsipy.configs import resolve_basis_path

                    resolved = resolve_basis_path(p, cfg_source=cfg_source)
                    if resolved != p:
                        input_cfg_file.set(sec, opt, resolved)
                        p = resolved
                except Exception:
                    pass

                if not os.path.exists(p):
                    missing_files.append(f"[{sec}] {opt} -> {p}")

            if missing_files:
                raise ConfigurationError(
                    "Basis-set file paths are missing or do not exist.\n\n" + "\n".join(f"- {m}" for m in missing_files)
                )

        if input_cfg_file.has_option('GLOBAL', 'dti_bval_cut'):
            try:
                bval_cut = int(input_cfg_file['GLOBAL']['dti_bval_cut'])
                if bval_cut < 0 or bval_cut > 20000:
                    raise ConfigurationError(
                        f"Invalid dti_bval_cut: {bval_cut}\n"
                        f"Should be between 0 and 20000 s/mm^2.\n"
                        f"Default: 1500 (standard low-b DTI cutoff).\n"
                        f"Higher values include additional shells; 15000 uses all b-values (nonstandard for DTI)."
                    )
            except ValueError:
                raise ConfigurationError(
                    f"Invalid dti_bval_cut value: must be an integer.\n"
                    f"Current value: '{input_cfg_file['GLOBAL']['dti_bval_cut']}'"
                )

        if engine == 'DTI' and input_cfg_file.has_section('DTI'):
            if input_cfg_file.has_option('DTI', 'dti_fit_method'):
                fit_method = str(input_cfg_file['DTI']['dti_fit_method']).upper().strip()
                if fit_method not in {'OLS', 'WLS', 'ADAM'}:
                    raise ConfigurationError(
                        f"Invalid dti_fit_method: {fit_method}\n" f"Valid values: OLS | WLS | ADAM"
                    )

            if input_cfg_file.has_option('DTI', 'dti_lr'):
                try:
                    lr = float(input_cfg_file['DTI']['dti_lr'])
                    if lr <= 0 or lr > 1:
                        raise ConfigurationError(
                            f"Invalid dti_lr: {lr}\n" f"Learning rate should be between 0 and 1."
                        )
                except ValueError:
                    raise ConfigurationError(
                        f"Invalid dti_lr value: must be a number.\n" f"Current value: '{input_cfg_file['DTI']['dti_lr']}'"
                    )

            if input_cfg_file.has_option('DTI', 'dti_epochs'):
                try:
                    epochs = int(input_cfg_file['DTI']['dti_epochs'])
                    if epochs < 1 or epochs > 10000:
                        raise ConfigurationError(
                            f"Invalid dti_epochs: {epochs}\n" f"Recommended range: 50-500 (ADAM only)"
                        )
                except ValueError:
                    raise ConfigurationError(
                        f"Invalid dti_epochs value: must be an integer.\n" f"Current value: '{input_cfg_file['DTI']['dti_epochs']}'"
                    )

        if engine == 'NODDI' and input_cfg_file.has_section('NODDI'):
            if input_cfg_file.has_option('NODDI', 'noddi_lr'):
                try:
                    lr = float(input_cfg_file['NODDI']['noddi_lr'])
                    if lr <= 0 or lr > 1:
                        raise ConfigurationError(
                            f"Invalid noddi_lr: {lr}\n"
                            f"Learning rate should be between 0 and 1.\n"
                            f"Typical values: 0.0001 - 0.01"
                        )
                except ValueError:
                    raise ConfigurationError(
                        f"Invalid noddi_lr value: must be a number.\n" f"Current value: '{input_cfg_file['NODDI']['noddi_lr']}'"
                    )

            if input_cfg_file.has_option('NODDI', 'noddi_epochs'):
                try:
                    epochs = int(input_cfg_file['NODDI']['noddi_epochs'])
                    if epochs < 10 or epochs > 10000:
                        raise ConfigurationError(
                            f"Invalid noddi_epochs: {epochs}\n"
                            f"Recommended range: 200-1000\n"
                            f"Very low (<10) may not converge; very high (>10000) wastes time"
                        )
                except ValueError:
                    raise ConfigurationError(
                        f"Invalid noddi_epochs value: must be an integer.\n" f"Current value: '{input_cfg_file['NODDI']['noddi_epochs']}'"
                    )

            if input_cfg_file.has_option('NODDI', 'noddi_d_ic'):
                try:
                    d_ic = float(input_cfg_file['NODDI']['noddi_d_ic'])
                    if d_ic <= 0 or d_ic > 5e-3:
                        raise ConfigurationError(
                            f"Invalid noddi_d_ic: {d_ic}\n"
                            f"Intra-cellular diffusivity (mm^2/s) should be in range (0, 5e-3].\n"
                            f"Typical value: 1.7e-3 mm^2/s"
                        )
                except ValueError:
                    raise ConfigurationError(
                        f"Invalid noddi_d_ic value: must be a number.\n" f"Current value: '{input_cfg_file['NODDI']['noddi_d_ic']}'"
                    )

            if input_cfg_file.has_option('NODDI', 'noddi_d_iso'):
                try:
                    d_iso = float(input_cfg_file['NODDI']['noddi_d_iso'])
                    if d_iso <= 0 or d_iso > 5e-3:
                        raise ConfigurationError(
                            f"Invalid noddi_d_iso: {d_iso}\n"
                            f"Isotropic diffusivity (mm^2/s) should be in range (0, 5e-3].\n"
                            f"Typical value: 3.0e-3 mm^2/s (free water)"
                        )
                except ValueError:
                    raise ConfigurationError(
                        f"Invalid noddi_d_iso value: must be a number.\n" f"Current value: '{input_cfg_file['NODDI']['noddi_d_iso']}'"
                    )

            # Optional device selection.
            # Defaults to CUDA if available; otherwise CPU.
            if input_cfg_file.has_section('DEVICE') and input_cfg_file.has_option('DEVICE', 'DEVICE'):
                dev = str(input_cfg_file.get('DEVICE', 'DEVICE', fallback='')).strip().lower()
                if dev not in {'', 'auto', 'cpu', 'cuda'}:
                    raise ConfigurationError(
                        f"Invalid [DEVICE] DEVICE: '{dev}'\n" "Valid values: auto | cpu | cuda"
                    )

        # DBSI/IA optimizer validation (Step 2 early stopping)
        if engine in {'DBSI', 'IA'} and input_cfg_file.has_section('OPTIMIZER'):
            if input_cfg_file.has_option('OPTIMIZER', 'step_2_patience'):
                try:
                    step_2_patience = int(input_cfg_file['OPTIMIZER']['step_2_patience'])
                    if step_2_patience < 1 or step_2_patience > 100000:
                        raise ConfigurationError(
                            f"Invalid step_2_patience: {step_2_patience}\n"
                            f"Must be an integer in [1, 100000].\n"
                            f"Default: 100"
                        )
                except ValueError:
                    raise ConfigurationError(
                        "Invalid step_2_patience value: must be an integer.\n"
                        f"Current value: '{input_cfg_file['OPTIMIZER']['step_2_patience']}'"
                    )

            if input_cfg_file.has_option('OPTIMIZER', 'step_2_min_delta'):
                try:
                    step_2_min_delta = float(input_cfg_file['OPTIMIZER']['step_2_min_delta'])
                    if step_2_min_delta < 0:
                        raise ConfigurationError(
                            f"Invalid step_2_min_delta: {step_2_min_delta}\n"
                            f"Must be >= 0.\n"
                            f"Default: 0.0"
                        )
                except ValueError:
                    raise ConfigurationError(
                        "Invalid step_2_min_delta value: must be a number.\n"
                        f"Current value: '{input_cfg_file['OPTIMIZER']['step_2_min_delta']}'"
                    )

        try:
            # DETAIL level (standard+) once logging is configured.
            from dbsipy.core.logfmt import DETAIL

            logging.getLogger().log(DETAIL, "✓ Configuration validation passed")
        except Exception:
            logging.info("✓ Configuration validation passed")

    def _setup_config(self, input_cfg_file) -> None:
        # Convert basis set paths between Windows and Linux automatically
        self._convert_basis_set_paths(input_cfg_file)

        # Input Parameters
        self.dwi_path = input_cfg_file['INPUT']['dwi_file']
        self.mask_path = input_cfg_file['INPUT'].get('mask_file', 'auto')
        # Mask semantics:
        # - missing key: default to 'auto' (legacy behavior)
        # - explicit 'auto': use median_otsu auto mask
        # - explicit empty value: use a minimal signal-based mask (drop only all-floor voxels)
        # - explicit none-like values: same as empty (n/a, na, none, n\a)
        if isinstance(self.mask_path, str):
            m = self.mask_path.strip().lower()
            if m in {'n/a', 'n\\a', 'na', 'none'}:
                self.mask_path = ''
        self.bval_path = input_cfg_file['INPUT']['bval_file']
        self.bvec_path = input_cfg_file['INPUT']['bvec_file']

        # Output mode (controls terminal verbosity + progress rendering).
        # Primary switch: [DEBUG] output_mode
        raw_output_mode = None
        try:
            raw_output_mode = input_cfg_file.get('DEBUG', 'output_mode', fallback=None)
        except Exception:
            raw_output_mode = None

        # Backwards compatibility (deprecated): legacy [DEBUG] verbose.
        if raw_output_mode is None:
            try:
                legacy_verbose = input_cfg_file.getboolean('DEBUG', 'verbose', fallback=False)
            except Exception:
                legacy_verbose = False
            if input_cfg_file.has_option('DEBUG', 'verbose'):
                logging.warning("Config key [DEBUG] verbose is deprecated; use [DEBUG] output_mode instead.")
            raw_output_mode = 'verbose' if bool(legacy_verbose) else 'standard'

        self._output_mode = self._normalize_output_mode(raw_output_mode)

        # Hidden INI-only feature flag (not documented): optionally emit the full
        # isotropic spectrum as a 4D map for DBSI/IA.
        # This is intentionally not exposed via CLI.
        try:
            self.emit_isotropic_spectrum = input_cfg_file.getboolean('DEBUG', 'emit_isotropic_spectrum', fallback=False)
        except Exception:
            self.emit_isotropic_spectrum = False

        # Keep legacy attribute for older call sites (derived from output_mode).
        self.verbose_flag = bool(self._output_mode in {'verbose', 'debug'})

        # Global Parameters
        self.ENGINE = input_cfg_file['GLOBAL']['model_engine']
        # Controls which parameter-map set(s) are allocated + saved.
        # When set to "expanded", the expanded map set includes the default maps.
        self.output_map_set = self._normalize_output_map_set(input_cfg_file['GLOBAL'].get('output_map_set', None))
        # DTI b-value cutoff (used for the optional DTI-for-comparison maps).
        # Default is a standard low-b cutoff; users can raise it to include more shells.
        self.dti_bval_cutoff = int(input_cfg_file['GLOBAL']['dti_bval_cut']) if input_cfg_file.has_option('GLOBAL', 'dti_bval_cut') else 1500

        # Signal normalization semantics.
        # Historically DBSI/IA operate on attenuation (signal divided by per-voxel max).
        raw_norm = input_cfg_file['GLOBAL'].get('signal_normalization', None)
        norm_mode = self._normalize_signal_normalization(raw_norm)
        if norm_mode == 'auto':
            if self.ENGINE in {'DBSI', 'IA', 'NODDI'}:
                norm_mode = 'max'
            else:
                norm_mode = 'none'
        self.signal_normalization = norm_mode

        # Optional voxelwise S0 scaling parameter (where implemented).
        self.learnable_s0 = input_cfg_file.getboolean('GLOBAL', 'learnable_s0', fallback=False)

        # Engine-specific parameters
        if self.ENGINE == 'DTI':
            dti_section = input_cfg_file['DTI'] if input_cfg_file.has_section('DTI') else {}
            self.dti_fit_method = str(dti_section.get('dti_fit_method', 'WLS')).upper()
            self.dti_lr = float(dti_section.get('dti_lr', 0.01))
            self.dti_epochs = int(dti_section.get('dti_epochs', 50))
            self.DTI_OPTIMIZER_ARGS = {'lr': self.dti_lr, 'epochs': self.dti_epochs}
            self.NODDI_OPTIMIZER_ARGS = None
        elif self.ENGINE == 'NODDI':
            # NODDI Parameters
            self.noddi_lr = float(input_cfg_file['NODDI']['noddi_lr']) if input_cfg_file.has_option('NODDI', 'noddi_lr') else 0.001
            self.noddi_epochs = int(input_cfg_file['NODDI']['noddi_epochs']) if input_cfg_file.has_option('NODDI', 'noddi_epochs') else 100
            self.noddi_batch_size = (
                int(input_cfg_file['NODDI']['noddi_batch_size'])
                if input_cfg_file.has_option('NODDI', 'noddi_batch_size')
                else None
            )
            self.noddi_d_ic = float(input_cfg_file['NODDI']['noddi_d_ic']) if input_cfg_file.has_option('NODDI', 'noddi_d_ic') else 1.7e-3
            self.noddi_d_iso = float(input_cfg_file['NODDI']['noddi_d_iso']) if input_cfg_file.has_option('NODDI', 'noddi_d_iso') else 3.0e-3
            self.noddi_use_tortuosity = input_cfg_file.getboolean('NODDI', 'noddi_use_tortuosity') if input_cfg_file.has_option('NODDI', 'noddi_use_tortuosity') else True
            self.NODDI_OPTIMIZER_ARGS = {
                'lr': self.noddi_lr,
                'epochs': self.noddi_epochs,
                'batch_size': self.noddi_batch_size,
                'd_ic': self.noddi_d_ic,
                'd_iso': self.noddi_d_iso,
                'use_tortuosity': self.noddi_use_tortuosity,
            }
        else:
            # DBSI/IA Parameters
            self.NODDI_OPTIMIZER_ARGS = None

            # Global DBSI/IA Parameters
            # `weight_threshold` is a fixed constant (not user-configurable).
            # If older configs specify it, ignore the value.
            self.weight_threshold = 0.003
            self.max_group_number = int(input_cfg_file['GLOBAL']['max_group_number'])
            self.fiber_threshold = float(input_cfg_file['GLOBAL']['fiber_threshold'])

            # Optimizer Arguments for DBSI/IA
            # Defaults are chosen to match shipped template configs (paper benchmarks):
            # Step 1 (orientation fitting) is fast, Step 2 (NNLS) is slower and benefits from more epochs.
            self.step_1_LR = float(input_cfg_file['OPTIMIZER']['step_1_lr']) if input_cfg_file.has_option('OPTIMIZER', 'step_1_lr') else 0.001
            self.step_1_epochs = int(input_cfg_file['OPTIMIZER']['step_1_epochs']) if input_cfg_file.has_option('OPTIMIZER', 'step_1_epochs') else 100
            self.step_1_loss = input_cfg_file['OPTIMIZER']['step_1_loss_fn'] if input_cfg_file.has_option('OPTIMIZER', 'step_1_loss_fn') else 'mse'

            self.step_2_LR = float(input_cfg_file['OPTIMIZER']['step_2_lr']) if input_cfg_file.has_option('OPTIMIZER', 'step_2_lr') else 0.001
            self.step_2_epochs = int(input_cfg_file['OPTIMIZER']['step_2_epochs']) if input_cfg_file.has_option('OPTIMIZER', 'step_2_epochs') else 250
            self.step_2_loss = input_cfg_file['OPTIMIZER']['step_2_loss_fn'] if input_cfg_file.has_option('OPTIMIZER', 'step_2_loss_fn') else 'mse'

            # Step 2 early stopping knobs (aggressive defaults for faster convergence)
            self.step_2_patience = int(input_cfg_file['OPTIMIZER']['step_2_patience']) if input_cfg_file.has_option('OPTIMIZER', 'step_2_patience') else 10
            self.step_2_min_delta = float(input_cfg_file['OPTIMIZER']['step_2_min_delta']) if input_cfg_file.has_option('OPTIMIZER', 'step_2_min_delta') else 1e-5

            self.STEP_1_OPTIMIZER_ARGS = {
                'optimizer': torch.optim.Adam,
                'lr': self.step_1_LR,
                'epochs': self.step_1_epochs,
                'loss': self.step_1_loss,
                'alpha': self.step_1_LR,
            }

            self.STEP_2_OPTIMIZER_ARGS = {
                'optimizer': torch.optim.Adam,
                'lr': self.step_2_LR,
                'epochs': self.step_2_epochs,
                'loss': self.step_2_loss,
                'alpha': self.step_2_LR,
                'patience': self.step_2_patience,
                'min_delta': self.step_2_min_delta,
            }

        # Step 1 Parameters (skip for DTI and NODDI engines)
        if self.ENGINE not in ['DTI', 'NODDI']:
            self.angle_threshold = int(input_cfg_file['STEP_1']['angle_threshold'])
            self.angle_basis = torch.from_numpy(
                np.squeeze(
                    np.asarray(
                        [line for line in csv.reader(open(input_cfg_file['STEP_1']['angle_basis'], mode='r'), delimiter=',')],
                        dtype=np.float32,
                    )
                )
            ).float()
            self.iso_basis = torch.from_numpy(
                np.squeeze(
                    np.asarray(
                        [line for line in csv.reader(open(input_cfg_file['STEP_1']['iso_basis'], mode='r'), delimiter=',')],
                        dtype=np.float32,
                    )
                )
            ).float()
            self.step_1_axial = torch.FloatTensor([float(input_cfg_file['STEP_1']['step_1_axial'])])
            self.step_1_radial = torch.FloatTensor([float(input_cfg_file['STEP_1']['step_1_radial'])])
            self.step_1_lambdas = torch.stack([self.step_1_axial, self.step_1_radial, self.step_1_radial], dim=1)

            # Step 2 Parameters
            self.step_2_axials = torch.from_numpy(
                np.squeeze(
                    np.asarray(
                        [line for line in csv.reader(open(input_cfg_file['STEP_2']['step_2_axials'], mode='r'), delimiter=',')],
                        dtype=np.float32,
                    )
                )
            ).float()
            self.step_2_radials = torch.from_numpy(
                np.squeeze(
                    np.asarray(
                        [line for line in csv.reader(open(input_cfg_file['STEP_2']['step_2_radials'], mode='r'), delimiter=',')],
                        dtype=np.float32,
                    )
                )
            ).float()
            self.intra_threshold = 1e3 * float(input_cfg_file['STEP_2']['intra_threshold'])
            self.step_2_lambdas = 1e-3 * torch.stack([self.step_2_axials, self.step_2_radials, self.step_2_radials], dim=1)

            # Isotropic Parameters
            self.restricted_threshold = float(input_cfg_file['ISOTROPIC']['restricted_threshold'])
            self.water_threshold = float(input_cfg_file['ISOTROPIC']['free_water_threshold'])
            try:
                self.highly_restricted_threshold = float(input_cfg_file['ISOTROPIC']['highly_restricted_threshold'])
                self.four_iso = True
                self.highly_restricted_inds = self.iso_basis <= self.highly_restricted_threshold
                self.restricted_inds = torch.logical_and(
                    self.iso_basis <= self.restricted_threshold, self.iso_basis > self.highly_restricted_threshold
                )
                self.water_inds = self.iso_basis >= self.water_threshold
                self.hindered_inds = torch.logical_and(
                    self.iso_basis > self.restricted_threshold, self.iso_basis < self.water_threshold
                )

                self.DEFAULT_ISOTROPIC_CUTS = {
                    'highly_restricted': self.highly_restricted_inds,
                    'restricted': self.restricted_inds,
                    'hindered': self.hindered_inds,
                    'water': self.water_inds,
                    'isotropic': torch.ones(self.restricted_inds.shape, dtype=torch.bool),
                }
            except (KeyError, ValueError, configparser.NoOptionError):
                self.highly_restricted_threshold = 0
                self.four_iso = False
                self.restricted_inds = self.iso_basis <= self.restricted_threshold
                self.water_inds = self.iso_basis >= self.water_threshold
                self.hindered_inds = torch.logical_and(
                    self.iso_basis > self.restricted_threshold, self.iso_basis < self.water_threshold
                )
                self.DEFAULT_ISOTROPIC_CUTS = {
                    'restricted': self.restricted_inds,
                    'hindered': self.hindered_inds,
                    'water': self.water_inds,
                    'isotropic': torch.ones(self.restricted_inds.shape, dtype=torch.bool),
                }
        else:
            # DTI and NODDI engines don't need these parameters, set defaults
            self.four_iso = False

        if self.ENGINE == 'IA':
            self.DEFAULT_FIBER_CUTS = {
                'IA': self.step_2_radials <= self.intra_threshold,
                'EA': self.step_2_radials > self.intra_threshold,
                'fiber': torch.ones(self.step_2_radials.shape, dtype=torch.bool),
            }
        elif self.ENGINE == 'DBSI':
            self.DEFAULT_FIBER_CUTS = {
                'fiber': torch.ones(self.step_2_radials.shape, dtype=torch.bool),
            }
        else:  # DTI/NODDI
            self.DEFAULT_FIBER_CUTS = None

        # Computing
        default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        cfg_device = None
        cfg_host = None
        try:
            if input_cfg_file.has_section('DEVICE'):
                cfg_device = input_cfg_file.get('DEVICE', 'DEVICE', fallback=None)
                cfg_host = input_cfg_file.get('DEVICE', 'HOST', fallback=None)
        except Exception:
            cfg_device = None
            cfg_host = None

        device_norm = (str(cfg_device).strip().lower() if cfg_device is not None else '')
        if device_norm in {'', 'auto'}:
            device_norm = default_device
        elif device_norm == 'cuda':
            if not torch.cuda.is_available():
                logging.warning("[DEVICE] DEVICE=cuda requested but CUDA is not available; falling back to cpu")
                device_norm = 'cpu'
        elif device_norm == 'cpu':
            device_norm = 'cpu'
        else:
            device_norm = default_device

        host_norm = (str(cfg_host).strip().lower() if cfg_host is not None else '')
        if host_norm in {'', 'auto'}:
            host_norm = 'cpu'
        if host_norm != 'cpu':
            logging.warning("[DEVICE] HOST=%s is not supported; using cpu", host_norm)
            host_norm = 'cpu'

        self.DEVICE = device_norm
        self.HOST = host_norm

        return
