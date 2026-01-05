__author__ = "Kainen Utt"
__credits__ = ["Kainen Utt", "Jacob Blum", "Yong Wang", "Sheng-Kwei (Victor) Song"]
from dbsipy._version import __version__
__maintainer__ = "Kainen Utt"
__email__ = "k.l.utt@wustl.edu"
__status__ = "Still Under Development"


import torch 
import numpy as np
import time
import nibabel as nb 


from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    # Only for type hints; keep runtime import headless-safe.
    from tkinter import Tk

import os 
import sys
import platform
import subprocess
import importlib
import shlex
from pathlib import Path
import logging
import configparser
from datetime import datetime

from dbsipy.core.logfmt import log_banner

from dbsipy.core import utils
from dbsipy.maps.dti_maps import default_dti_parameter_maps, expanded_dti_parameter_maps
from dbsipy.maps.dbsi_maps import default_dbsi_parameter_maps, expanded_dbsi_parameter_maps, four_seg_iso_maps, three_seg_iso_maps
from dbsipy.maps.dbsi_ia_maps import default_dbsi_ia_parameter_maps, expanded_dbsi_ia_parameter_maps
from dbsipy.maps.noddi_maps import SCALAR_MAPS as noddi_scalar_maps, VECTOR_MAPS as noddi_vector_maps

# Default PyTorch Datatype is float32. 
MIN_POSITIVE_SIGNAL = np.finfo(np.float32).eps

# NODDI parameter maps (scalars + directional maps + DTI for comparison)
default_noddi_parameter_maps = {**{k: 'scalar' for k in noddi_scalar_maps.keys()},
              **{k: '3-vector' for k in noddi_vector_maps.keys()},
              **default_dti_parameter_maps}  # Include DTI maps for comparison

expanded_noddi_parameter_maps = {**{k: 'scalar' for k in noddi_scalar_maps.keys()},
              **{k: '3-vector' for k in noddi_vector_maps.keys()},
              # Expanded outputs should include the default DTI set plus the expanded extras.
              **default_dti_parameter_maps,
              **expanded_dti_parameter_maps}  # Include DTI maps for comparison

default_parameter_maps = {
                          'DTI' : default_dti_parameter_maps,
                          # DBSI outputs + DTI for comparison
                          'DBSI': {**default_dbsi_parameter_maps, **default_dti_parameter_maps},
                          # IA outputs + DTI for comparison
                          'IA'  : {**default_dbsi_ia_parameter_maps, **default_dti_parameter_maps},
                          'NODDI': default_noddi_parameter_maps
                         }

expanded_parameter_maps = {
                          # "expanded" maps are additive extras; include default + expanded.
                          'DTI' : {**default_dti_parameter_maps, **expanded_dti_parameter_maps},
                          # DBSI expanded outputs + DTI for comparison
                          'DBSI': {
                              **default_dbsi_parameter_maps,
                              **expanded_dbsi_parameter_maps,
                              **default_dti_parameter_maps,
                              **expanded_dti_parameter_maps,
                          },
                          # IA expanded outputs + DTI for comparison
                          'IA'  : {
                              **default_dbsi_ia_parameter_maps,
                              **expanded_dbsi_ia_parameter_maps,
                              **default_dti_parameter_maps,
                              **expanded_dti_parameter_maps,
                          },
                          'NODDI': expanded_noddi_parameter_maps
                         }

isotropic_parameter_maps = {True: four_seg_iso_maps,
                            False: three_seg_iso_maps}

from dbsipy.core.validation import DataError, validate_tensor
from dbsipy.core.signal import normalize_signal
from dbsipy.core.io import load_dwi_nifti, load_bvals_bvecs, mask_dwi, save_auto_mask_nifti
from dbsipy.core.configuration import configuration
from dbsipy.dti.engine import run_dti
from dbsipy.noddi.engine import run_noddi
from dbsipy.dbsi.engine import run_dbsi
from dbsipy.dbsi_ia.engine import run_ia


class _DBSIFormatter(logging.Formatter):
    """Consistent, readable console/file formatting.

    - INFO:    "DBSI: <message>"
    - WARNING: "DBSI [WARNING]: <message>"
    - ERROR:   "DBSI [ERROR]: <message>"
    """

    def format(self, record: logging.LogRecord) -> str:
        prefix = "DBSI"
        if record.levelno == logging.INFO:
            return f"{prefix}: {record.getMessage()}"
        return f"{prefix} [{record.levelname}]: {record.getMessage()}"

class DBSIpy:
    def __init__(self, cfg_file: Type[configparser.ConfigParser]) -> None:
        self.configuration = configuration(cfg_file)
        self.params = utils.ParamStoreDict()
        self._timings: dict[str, float] = {}
        self._flags: dict[str, bool] = {}
        self._git_commit: str | None = None
        self._argv_str: str | None = None
        self._total_runtime_s: float | None = None
        self._run_started_utc: str | None = None
        self._run_finished_utc: str | None = None
        self._rng: dict[str, object] = {}
        self.configure_logging()
        return

    def _configure_reproducibility(self) -> None:
        """Best-effort determinism and RNG seeding.

        Controlled via env vars:
        - DBSIPY_SEED: integer seed applied to numpy + torch (+ torch.cuda if available)
        - DBSIPY_DETERMINISTIC=1: enable best-effort deterministic flags

        This is intentionally best-effort and should not abort runs.
        """
        rng: dict[str, object] = {
            'seed_env': os.environ.get('DBSIPY_SEED', None),
            'deterministic_env': os.environ.get('DBSIPY_DETERMINISTIC', '0'),
        }

        # Seed
        seed_val = None
        try:
            if rng['seed_env'] is not None and str(rng['seed_env']).strip() != '':
                seed_val = int(str(rng['seed_env']).strip())
        except Exception:
            seed_val = None

        if seed_val is not None:
            try:
                np.random.seed(seed_val)
            except Exception:
                pass
            try:
                torch.manual_seed(seed_val)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_val)
            except Exception:
                pass
            rng['seed'] = int(seed_val)

        # Determinism (best-effort)
        deterministic = (os.environ.get('DBSIPY_DETERMINISTIC', '0') == '1')
        rng['deterministic'] = bool(deterministic)
        if deterministic:
            try:
                # Warn-only so we don't hard-fail on a nondeterministic op.
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
            logging.info(
                "Deterministic mode enabled (best-effort). Some operations may remain nondeterministic; performance may be reduced."
            )

        try:
            rng['torch_initial_seed'] = int(torch.initial_seed())
        except Exception:
            pass
        self._rng = rng

    @staticmethod
    def _safe_version(mod_name: str) -> str:
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, '__version__', 'unknown')
        except Exception:
            return 'unavailable'

    def _write_run_manifest(self) -> None:
        """Write a provenance manifest alongside saved outputs.

        This is intentionally best-effort: failures should not abort the run.
        """
        try:
            from dbsipy.core.provenance import write_run_manifest

            write_run_manifest(self, dbsipy_version=__version__)
        except Exception:
            logging.exception('Failed to write run_manifest.json')

    def _log_banner(self, title: str) -> None:
        log_banner(title)

    def _write_final_config_snapshot(self) -> None:
        cfg = self.configuration.cfg_file

        for section in ['INPUT', 'GLOBAL', 'DEBUG']:
            if not cfg.has_section(section):
                cfg.add_section(section)

        cfg.set('GLOBAL', 'model_engine', str(self.configuration.ENGINE))
        cfg.set('GLOBAL', 'dti_bval_cut', str(self.configuration.dti_bval_cutoff))
        cfg.set('GLOBAL', 'output_map_set', str(self.configuration.output_map_set))
        try:
            cfg.set('GLOBAL', 'signal_normalization', str(getattr(self.configuration, 'signal_normalization', 'auto')))
            cfg.set('GLOBAL', 'learnable_s0', str(bool(getattr(self.configuration, 'learnable_s0', False))))
        except Exception:
            pass
        cfg.set('DEBUG', 'verbose', str(bool(self.configuration.verbose_flag)))

        # Best-effort provenance for troubleshooting.
        cfg_source = cfg.get('DEBUG', 'cfg_source', fallback=None)
        if cfg_source is not None:
            cfg.set('DEBUG', 'cfg_source', str(cfg_source))

        if self.configuration.ENGINE == 'DTI':
            if not cfg.has_section('DTI'):
                cfg.add_section('DTI')
            cfg.set('DTI', 'dti_fit_method', str(getattr(self.configuration, 'dti_fit_method', 'WLS')))
            cfg.set('DTI', 'dti_lr', str(getattr(self.configuration, 'dti_lr', 0.01)))
            cfg.set('DTI', 'dti_epochs', str(getattr(self.configuration, 'dti_epochs', 200)))

        if self.configuration.ENGINE == 'NODDI':
            if not cfg.has_section('NODDI'):
                cfg.add_section('NODDI')
            cfg.set('NODDI', 'noddi_lr', str(self.configuration.noddi_lr))
            cfg.set('NODDI', 'noddi_epochs', str(self.configuration.noddi_epochs))
            cfg.set('NODDI', 'noddi_d_ic', str(self.configuration.noddi_d_ic))
            cfg.set('NODDI', 'noddi_d_iso', str(self.configuration.noddi_d_iso))
            cfg.set('NODDI', 'noddi_use_tortuosity', str(self.configuration.noddi_use_tortuosity))

        snapshot_path = os.path.join(self.save_dir, 'config_final.ini')
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            cfg.write(f)

        logging.info(f"Config snapshot saved: {os.path.basename(snapshot_path)}")

    def _log_verbose_runtime_environment(self) -> None:
        logging.info(' ----------------------------- ')
        logging.info('      Runtime Environment      ')
        logging.info(' ----------------------------- ')
        logging.info(f"  DBSIpy Version   : {__version__}")
        logging.info(f"  Python           : {sys.version.split()[0]}")
        logging.info(f"  Platform         : {platform.platform()}")
        logging.info(f"  NumPy            : {self._safe_version('numpy')}")
        logging.info(f"  PyTorch          : {self._safe_version('torch')}")
        logging.info(f"  NiBabel          : {self._safe_version('nibabel')}")
        logging.info(f"  DIPY             : {self._safe_version('dipy')}")
        logging.info(f"  Working Dir      : {os.getcwd()}")

        try:
            argv_str = shlex.join(sys.argv)
        except Exception:
            argv_str = ' '.join(str(a) for a in sys.argv)
        self._argv_str = argv_str
        logging.info(f"  Command          : {argv_str}")
        logging.info(f"  CUDA Available   : {torch.cuda.is_available()}")
        logging.info(f"  Selected DEVICE  : {self.configuration.DEVICE}")
        if torch.cuda.is_available():
            try:
                logging.info(f"  CUDA Device      : {torch.cuda.get_device_name(0)}")
            except Exception:
                pass

        # Best-effort git SHA for reproducibility (works when running from a git checkout)
        try:
            project_root = Path(__file__).parent.parent.parent
            sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(project_root), stderr=subprocess.DEVNULL)
            self._git_commit = sha.decode().strip()
            logging.info(f"  Git Commit       : {self._git_commit}")
        except Exception:
            pass
    
    def configure_logging(self) -> None:
        
        DEFAULT_INPUT_PARAMETERS = [
                                    'STEP_1_OPTIMIZER_ARGS',
                                    'STEP_2_OPTIMIZER_ARGS',
                                    'dwi_path',
                                    'mask_path',
                                    'bval_path',
                                    'bvec_path',
                                    'weight_threshold',
                                    'angle_threshold',
                                    'max_group_number',
                                    'restricted_threshold',
                                    'water_threshold',
                                    'ENGINE',
                                    'DEVICE',
                                    'HOST',
                                    'bval_cutoff',
                                    'intra_threshold'
                                    ]
        
        time = datetime.now().strftime('%Y%m%d_%H%M')

        # Optional output-dir override for programmatic runs (e.g., benchmarking).
        # When unset, preserve historical behavior: save next to the DWI.
        base_dir = Path(self.configuration.dwi_path).parent.absolute()
        run_tag = None
        try:
            cfg = getattr(self.configuration, 'cfg_file', None)
            if cfg is not None and cfg.has_section('OUTPUT'):
                raw_save_dir = cfg.get('OUTPUT', 'save_dir', fallback=None)
                raw_run_tag = cfg.get('OUTPUT', 'run_tag', fallback=None)

                if raw_save_dir is not None and str(raw_save_dir).strip() != '':
                    save_dir = Path(str(raw_save_dir).strip())
                    if not save_dir.is_absolute():
                        cfg_source = None
                        try:
                            cfg_source = cfg.get('DEBUG', 'cfg_source', fallback=None)
                        except Exception:
                            cfg_source = None
                        if cfg_source:
                            save_dir = Path(str(cfg_source)).resolve().parent / save_dir
                        else:
                            save_dir = Path.cwd() / save_dir
                    base_dir = save_dir

                if raw_run_tag is not None and str(raw_run_tag).strip() != '':
                    # Keep filenames filesystem-friendly.
                    run_tag = ''.join(
                        (c if (c.isalnum() or c in {'-', '_'}) else '_') for c in str(raw_run_tag).strip()
                    ).strip('_')
        except Exception:
            base_dir = Path(self.configuration.dwi_path).parent.absolute()
            run_tag = None

        prefix = f"{time}_" + (f"{run_tag}_" if run_tag else "")

        if self.configuration.ENGINE == 'DTI':
            # DTI engine - simple output directory
            self.save_dir = os.path.join(
                                    base_dir,
                                    f'{prefix}DTI_Results'
                                )
        elif self.configuration.ENGINE == 'NODDI':
            # NODDI engine - simple output directory
            self.save_dir = os.path.join(
                                    base_dir,
                                    f'{prefix}NODDI_Results'
                                )
        elif self.configuration.ENGINE == 'IA': 
            if self.configuration.four_iso:
                self.save_dir = os.path.join(
                                        base_dir,
                                        f'{prefix}DBSI_{self.configuration.ENGINE}_Results_{1e3 * self.configuration.highly_restricted_threshold}_{1e3 * self.configuration.restricted_threshold}_{1e3 * self.configuration.restricted_threshold}_{1e3 * self.configuration.water_threshold}_{1e3 * self.configuration.water_threshold}'
                                    )
            else:
                self.save_dir = os.path.join(
                                        base_dir,
                                        f'{prefix}DBSI_{self.configuration.ENGINE}_Results_{1e3 * self.configuration.restricted_threshold}_{1e3 * self.configuration.restricted_threshold}_{1e3 * self.configuration.water_threshold}_{1e3 * self.configuration.water_threshold}'
                                    )
        else:
            # DBSI engine
            # Historically this created folders like "..._DBSI_DBSI_Results..." when ENGINE=="DBSI".
            # Keep "DBSI_" prefix only once.
            if self.configuration.ENGINE == 'DBSI':
                engine_label = 'DBSI'
            else:
                engine_label = f'DBSI_{self.configuration.ENGINE}'

            self.save_dir = os.path.join(
                                    base_dir,
                                    f'{prefix}{engine_label}_Results_{1e3 * self.configuration.restricted_threshold}_{1e3 * self.configuration.restricted_threshold}_{1e3 * self.configuration.water_threshold}_{1e3 * self.configuration.water_threshold}'
                                )

        if not os.path.exists(self.save_dir): os.mkdir(self.save_dir)
        
        #### Configure Log File ####
        log_file = os.path.join(self.save_dir, 'log')
        
        # Configure logging with consistent formatting to both file and console.
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(_DBSIFormatter())

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(_DBSIFormatter())

        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler],
            force=True,  # Python 3.8+: force reconfiguration
        )

        # Save a snapshot of the fully resolved config into the results folder.
        # This is especially useful for GUI runs where configs are edited interactively.
        self._write_final_config_snapshot()

        # Add richer environment/version details only when verbose is enabled.
        if self.configuration.verbose_flag:
            self._log_verbose_runtime_environment()
     
        self._log_banner('Input Parameters')
        if self.configuration.verbose_flag and self.configuration.ENGINE in ['DBSI', 'IA']:
            logging.info(' ----------------------------- ')
            logging.info('      Optimizer Arguments      ')
            logging.info(' ----------------------------- ')
            logging.info(f"  Step 1 Learning Rate: {self.configuration.step_1_LR}")
            logging.info(f"  Step 1 Epochs       : {self.configuration.step_1_epochs}")
            logging.info(f"  Step 1 Loss Function: {self.configuration.step_1_loss}")
            logging.info(f"  Step 2 Learning Rate: {self.configuration.step_2_LR}")
            logging.info(f"  Step 2 Epochs       : {self.configuration.step_2_epochs}")
            logging.info(f"  Step 2 Loss Function: {self.configuration.step_2_loss}")
        logging.info(' ----------------------------- ')
        logging.info('          Input Files          ')
        logging.info(' ----------------------------- ')
        logging.info(f"  DWI File     : {os.path.split(self.configuration.dwi_path)[1]}")
        logging.info(f"  Mask File    : {os.path.split(self.configuration.mask_path)[1]}")
        logging.info(f"  B-value File : {os.path.split(self.configuration.bval_path)[1]}")
        logging.info(f"  B-vector File: {os.path.split(self.configuration.bvec_path)[1]}")

        logging.info(
            f"  Run Config   : engine={self.configuration.ENGINE}, output_maps={self.configuration.output_map_set}, "
            f"device={self.configuration.DEVICE}, dti_bval_cut={self.configuration.dti_bval_cutoff}, "
            f"verbose={bool(self.configuration.verbose_flag)}, diagnostics={bool(getattr(self.configuration, 'diagnostics_enabled', False))}"
        )
        if self.configuration.verbose_flag:
            logging.info(' ----------------------------- ')
            logging.info('       Global Parameters       ')
            logging.info(' ----------------------------- ')
            logging.info(f"  Max. B-value Used: {self.configuration.dti_bval_cutoff}")
            logging.info(f"  Model Engine     : {self.configuration.ENGINE}")

            if self.configuration.ENGINE == 'DTI':
                logging.info(' ----------------------------- ')
                logging.info('    DTI Fit Parameters         ')
                logging.info(' ----------------------------- ')
                logging.info(f"  Fit Method       : {getattr(self.configuration, 'dti_fit_method', 'WLS')}")
                logging.info(f"  Learning Rate    : {getattr(self.configuration, 'dti_lr', 0.01)}")
                logging.info(f"  Epochs           : {getattr(self.configuration, 'dti_epochs', 200)}")
            elif self.configuration.ENGINE == 'NODDI':
                # NODDI-specific parameters
                logging.info(' ----------------------------- ')
                logging.info('   NODDI Optimizer Arguments   ')
                logging.info(' ----------------------------- ')
                logging.info(f"  Learning Rate         : {self.configuration.noddi_lr}")
                logging.info(f"  Epochs                : {self.configuration.noddi_epochs}")
                logging.info(f"  Intra-cellular d      : {self.configuration.noddi_d_ic*1e3} um^2/ms")
                logging.info(f"  Isotropic d           : {self.configuration.noddi_d_iso*1e3} um^2/ms")
                logging.info(f"  Tortuosity constraint : {self.configuration.noddi_use_tortuosity}")
            else:
                # DBSI/IA-specific parameters
                logging.info(f"  Weight Threshold : {self.configuration.weight_threshold}")
                logging.info(f"  Max. Fiber Number: {self.configuration.max_group_number}")
                logging.info(' ----------------------------- ')
                logging.info('   Step 1 Fitting Parameters   ')
                logging.info(' ----------------------------- ')
                logging.info(f"  Angle Threshold          : {self.configuration.angle_threshold}")
                logging.info(f"  Fiber Weight Threshold   : {self.configuration.fiber_threshold}")
                logging.info(f"  Angle Basis Set          : {self.configuration.cfg_file['STEP_1']['angle_basis']}")
                logging.info(f"  Isotropic Basis Set      : {self.configuration.cfg_file['STEP_1']['iso_basis']}")
                logging.info(f"  Step 1 Axial Diffusivity : {round(float(self.configuration.step_1_axial)*1e3,3)} um^2/ms")
                logging.info(f"  Step 1 Radial Diffusivity: {round(float(self.configuration.step_1_radial)*1e3,3)} um^2/ms")
                logging.info(' ----------------------------- ')
                logging.info('   Step 2 Fitting Parameters   ')
                logging.info(' ----------------------------- ')
                logging.info(f"  Step 2 Axial Basis Set   : {self.configuration.cfg_file['STEP_2']['step_2_axials']}")
                logging.info(f"  Step 2 Radial Basis Set  : {self.configuration.cfg_file['STEP_2']['step_2_radials']}")
                if self.configuration.ENGINE == 'IA':
                    logging.info(f"  Intra-Axonal RD Threshold: {self.configuration.intra_threshold} um^2/ms")
                logging.info(' ----------------------------- ')
                logging.info(' Isotropic Spectrum Partitions ')
                logging.info(' ----------------------------- ')
                if self.configuration.four_iso:
                    logging.info(f"  Highly-Restricted Threshold: <{self.configuration.highly_restricted_threshold*1e3} um^2/ms")
                logging.info(f"  Restricted Threshold       : <{self.configuration.restricted_threshold*1e3} um^2/ms")
                logging.info(f"  Free Water Threshold       : >{self.configuration.water_threshold*1e3} um^2/ms")
        return
    
    def load(self) -> None:
        # ------------------------------------------------------------------------------- #
        #                              Load Diffusion Data                                #
        # ------------------------------------------------------------------------------- #
  
        # Cull voxels with signals below PyTorch's machine limits for float32 types.
        self.dwi, self.header, self.affine = load_dwi_nifti(self.configuration.dwi_path)

        # Record original 4D shape for provenance/manifest reporting.
        try:
            self.dwi_nifti_shape = tuple(int(x) for x in self.dwi.shape)
        except Exception:
            self.dwi_nifti_shape = None

        self.bvals, self.bvecs = load_bvals_bvecs(self.configuration.bval_path, self.configuration.bvec_path)

        assert self.bvals.shape[0] == self.bvecs.shape[0], (
            "Please ensure that .bval and .bvec file have the number of volumes as the first dimension"
        )
        logging.info(
            f"Loaded DWI: shape={self.dwi.shape[0]} x {self.dwi.shape[1]} x {self.dwi.shape[2]} x {self.dwi.shape[3]}, "
            f"volumes={len(self.bvals)}"
        )

        if self.configuration.diagnostics_enabled:
            try:
                logging.info(f"Diagnostics: ENGINE={self.configuration.ENGINE}, DEVICE={self.configuration.DEVICE}, HOST={getattr(self.configuration, 'HOST', 'unknown')}")
                if torch.cuda.is_available():
                    logging.info(f"Diagnostics: CUDA available, current_device={torch.cuda.current_device()}, name={torch.cuda.get_device_name(torch.cuda.current_device())}")
            except Exception:
                pass
        

        # ------------------------------------------------------------------------------- #
        #                                   Apply Mask                                    #
        # ------------------------------------------------------------------------------- #

        mask, dwi_masked, mask_source, spatial_dims, vol_idx, non_trivial_signal_mask = mask_dwi(
            self.dwi,
            bvals=self.bvals,
            mask_path=self.configuration.mask_path,
        )

        # If auto-masking was used, persist the generated mask next to the input DWI.
        if str(self.configuration.mask_path).strip().lower() == 'auto' and mask_source == 'auto':
            try:
                out_mask = save_auto_mask_nifti(
                    mask,
                    dwi_path=self.configuration.dwi_path,
                    affine=self.affine,
                    header=self.header,
                )
                logging.info(f"Auto mask saved: {out_mask}")
            except Exception as e:
                logging.warning(f"Could not save auto mask next to DWI: {e}")

        self.configuration.spatial_dims = spatial_dims
        if self.configuration.diagnostics_enabled:
            logging.info(f'Diagnostics: volume_index={vol_idx}')
            try:
                logging.info(
                    f"Diagnostics: non_trivial_signal_mask_shape="
                    f"{non_trivial_signal_mask.shape[0]} x {non_trivial_signal_mask.shape[1]} x {non_trivial_signal_mask.shape[2]}"
                )
            except Exception:
                pass

        self.dwi = torch.from_numpy(dwi_masked).float()
        self.dwi = self.dwi.reshape(-1, self.bvals.shape[0])
        self.mask_source = mask_source
        logging.info(
            f"Mask applied: source={mask_source}, masked_voxels={self.dwi.shape[0]:,}, volumes={self.dwi.shape[1]}"
        )
        if self.configuration.diagnostics_enabled:
            logging.info(f"Diagnostics: linearized_dwi_shape={self.dwi.shape[0]} x {self.dwi.shape[1]}")
        
        # Validate DWI data after loading
        try:
            validate_tensor(self.dwi, "DWI signal after masking", allow_negative=False, allow_inf=False)
            if self.configuration.diagnostics_enabled:
                logging.info("Diagnostics: DWI data validation passed")
        except DataError as e:
            logging.error(f"✗ DWI data validation failed:\n{e}")
            raise

        self._apply_signal_normalization()
        
        # Validate normalized signal
        try:
            validate_tensor(self.dwi, "Normalized DWI signal", allow_negative=False, allow_inf=False)
            if self.configuration.diagnostics_enabled:
                logging.info("Diagnostics: Normalized DWI signal validation passed")
        except DataError as e:
            logging.error(f"✗ Normalized signal validation failed:\n{e}")
            raise

        # Set mask and linear_dims attributes
        self.configuration.mask = mask
        self.configuration.linear_dims = self.dwi.shape[0] # These are masked dimensions

        if self.configuration.diagnostics_enabled:
            try:
                logging.info(f"Diagnostics: spatial_dims={self.configuration.spatial_dims}, masked_voxels={self.configuration.linear_dims:,}")
                logging.info(f"Diagnostics: bvals={int(self.bvals.shape[0])}, non_trivial_voxels={(~(self.dwi_raw == 0).all(dim=1)).sum().item():,}")
                if self.signal_scale_stats is not None:
                    logging.info(
                        "Diagnostics: signal_normalization=%s, scale_min=%.6g, scale_median=%.6g, scale_max=%.6g",
                        self.signal_normalization_mode,
                        self.signal_scale_stats['min'],
                        self.signal_scale_stats['median'],
                        self.signal_scale_stats['max'],
                    )
            except Exception:
                pass

        # Prepare the Parameter Maps 
        self.parameter_maps = self._prepare_parameter_maps()

        self._select_step2_target()

        return 

    def _apply_signal_normalization(self) -> None:
        """Apply configured signal normalization in-place.

        Expects `self.dwi` to be a (n_voxels, n_volumes) torch.FloatTensor and
        `self.bvals` to be available.

        Sets:
        - self.dwi_raw (copy of pre-normalization)
        - self.signal_normalization_mode
        - self.s0_est (per-voxel scale estimate)
        - self.signal_scale_stats (min/median/max over nonzero voxels)
        """
        raw_mode = str(getattr(self.configuration, 'signal_normalization', 'max'))
        engine = getattr(self.configuration, 'ENGINE', None)

        dwi_raw, dwi_norm, scale, scale_stats, mode_used = normalize_signal(
            self.dwi,
            self.bvals,
            mode=raw_mode,
            engine=str(engine) if engine is not None else None,
        )

        self.dwi_raw = dwi_raw
        self.dwi = dwi_norm
        self.signal_normalization_mode = mode_used
        self.s0_est = scale
        self.signal_scale_stats = scale_stats

    def _select_step2_target(self) -> None:
        """Select Step 2 fitting target (attenuation vs raw signal)."""
        # DBSI/IA historically fit attenuation. If learnable_s0 is enabled, Step 2 can
        # optionally fit raw signal using a voxelwise S0 scale parameter.
        if getattr(self.configuration, 'ENGINE', None) in {'DBSI', 'IA'} and bool(getattr(self.configuration, 'learnable_s0', False)):
            self.dwi_step2_target = self.dwi_raw
        else:
            self.dwi_step2_target = self.dwi
        
    def _prepare_parameter_maps(self) -> None:
        # Type[utils.parameter_map].__init__() requires pmap shapes in the flattened (linear) dimension, as well as the spatial dimension
        param_shapes = {
            'scalar'   : [(self.configuration.linear_dims, 1), (self.configuration.spatial_dims)],
            '3-vector' : [(self.configuration.linear_dims, 3), (self.configuration.spatial_dims) +(3,)]
        }
        
        # Add spectrum shape only for DBSI/IA engines that have iso_basis
        if self.configuration.ENGINE in ['DBSI', 'IA']:
            param_shapes['spectrum'] = [(self.configuration.linear_dims, len(self.configuration.iso_basis)), 
                                        (self.configuration.spatial_dims) +(len(self.configuration.iso_basis),)]
            
        def _alloc_map(map_name: str, map_shape: str) -> None:
            self.params[map_name] = utils.parameter_map(
                pmap_name=map_name,
                pmap_shapes=param_shapes[map_shape],
                mask=self.configuration.mask,
                device=self.configuration.DEVICE,
                header=self.header,
                affine=self.affine,
            )

        # Initialize parameter maps based on configuration.
        # Semantics:
        #   - default: allocate default maps only
        #   - expanded: allocate default maps + expanded extras

        allocated_map_defs: dict[str, str] = {}

        if self.configuration.output_map_set == 'default':
            for (map_name, map_shape) in default_parameter_maps[self.configuration.ENGINE].items():
                _alloc_map(map_name, map_shape)
                allocated_map_defs[map_name] = map_shape

        if self.configuration.output_map_set == 'expanded':
            for (map_name, map_shape) in default_parameter_maps[self.configuration.ENGINE].items():
                if map_name in self.params:
                    continue
                _alloc_map(map_name, map_shape)
                allocated_map_defs[map_name] = map_shape
            for (map_name, map_shape) in expanded_parameter_maps[self.configuration.ENGINE].items():
                _alloc_map(map_name, map_shape)
                allocated_map_defs[map_name] = map_shape

        # Learnable S0 output: when enabled (DBSI/IA Step 2), emit a voxelwise S0 map.
        if self.configuration.ENGINE in {'DBSI', 'IA'} and bool(getattr(self.configuration, 'learnable_s0', False)):
            if 's0_map' not in self.params:
                _alloc_map('s0_map', 'scalar')

        # Multi-fiber support (DBSI/IA): allocate additional fiber_{i}d_* maps.
        # Internal convention:
        #   - fiber_0d_* for the first fiber
        #   - fiber_1d_* for the second fiber, ...
        # Disk naming is handled by utils.legacy_fiber_save_name() in save().
        if self.configuration.ENGINE in {'DBSI', 'IA'}:
            try:
                max_fibers = int(getattr(self.configuration, 'max_group_number', 1) or 1)
            except Exception:
                max_fibers = 1

            if max_fibers > 1:
                for base_name, base_shape in list(allocated_map_defs.items()):
                    if not isinstance(base_name, str) or not base_name.startswith('fiber_0d_'):
                        continue

                    suffix = base_name[len('fiber_0d_'):]
                    for fiber_i in range(1, max_fibers):
                        new_name = f'fiber_{fiber_i}d_{suffix}'
                        if new_name in self.params:
                            continue
                        _alloc_map(new_name, base_shape)

                # Fraction-weighted aggregate fiber maps (only for multi-fiber runs)
                # These are internal names (saved as-is):
                #   - fiber_total_fraction: sum of per-fiber fractions
                #   - fiber_agg_*: fraction-weighted mean across fibers, normalized by total fiber fraction
                # Allocate only when default maps are present (keeps single-fiber outputs unchanged).
                if self.configuration.output_map_set in {'default', 'expanded'}:
                    for agg_name in (
                        'fiber_total_fraction',
                        'fiber_agg_fa',
                        'fiber_agg_axial',
                        'fiber_agg_radial',
                        'fiber_agg_adc',
                    ):
                        if agg_name not in self.params:
                            _alloc_map(agg_name, 'scalar')

                    # IA/EA-specific aggregates (only meaningful for IA engine)
                    if self.configuration.ENGINE == 'IA':
                        for agg_name in (
                            'fiber_total_IA_fraction',
                            'fiber_total_EA_fraction',
                            'fiber_agg_IA_fa',
                            'fiber_agg_IA_axial',
                            'fiber_agg_IA_radial',
                            'fiber_agg_IA_adc',
                            'fiber_agg_EA_fa',
                            'fiber_agg_EA_axial',
                            'fiber_agg_EA_radial',
                            'fiber_agg_EA_adc',
                        ):
                            if agg_name not in self.params:
                                _alloc_map(agg_name, 'scalar')
            
        # Initialize the Isotropic Parameter Maps (DBSI/IA only)
        if self.configuration.ENGINE in ['DBSI', 'IA']:
            for (map_name, map_shape) in isotropic_parameter_maps[self.configuration.four_iso].items():
                self.params[map_name] = utils.parameter_map(pmap_name = map_name, 
                                                                     pmap_shapes= param_shapes[map_shape], 
                                                                     mask = self.configuration.mask,
                                                                     device = self.configuration.DEVICE,
                                                                     header = self.header,
                                                                     affine = self.affine
                                                                     ) 
       
    def calc(self) -> None:

        if self.configuration.ENGINE == 'DTI':
            run_dti(self)
            return

        if self.configuration.ENGINE == 'NODDI':
            run_noddi(self)
            return

        if self.configuration.ENGINE == 'DBSI':
            run_dbsi(self)
            return

        if self.configuration.ENGINE == 'IA':
            run_ia(self)
            return

        raise ValueError(
            f"Unknown engine '{self.configuration.ENGINE}'. Must be DBSI, IA, DTI, or NODDI."
        )
        
        return

    def save(self) -> None:
        ## Save Results 
        logging.info('Saving results to: {}'.format(self.save_dir))    
        os.makedirs(self.save_dir, exist_ok=True)

        # Legacy-compatible fiber map naming on disk.
        # Internal parameter keys remain unchanged (e.g., fiber_0d_adc), but the
        # saved filenames become:
        # - max_group_number <= 1: fiber_adc
        # - max_group_number  > 1: fiber_01_adc, fiber_02_adc, ...
        max_fibers = int(getattr(self.configuration, 'max_group_number', 1) or 1)

        for pmap in self.params.values():
            original_name = getattr(pmap, 'pmap_name', None)
            save_name = utils.legacy_fiber_save_name(original_name, max_fibers)

            if save_name != original_name and original_name is not None:
                try:
                    pmap.pmap_name = save_name
                    pmap._save(self.save_dir)
                finally:
                    pmap.pmap_name = original_name
            else:
                pmap._save(self.save_dir)
        
        # Save configuration file to results directory
        config_save_path = os.path.join(self.save_dir, 'analysis_config.ini')
        with open(config_save_path, 'w') as configfile:
            self.configuration.cfg_file.write(configfile)
        logging.info(f'Configuration saved to: {config_save_path}')
        
        return
    
    def __call__(self) -> None:

        engine = str(self.configuration.ENGINE).upper()
        engine_display = 'DBSI-IA' if engine == 'IA' else engine
        self._log_banner(f'Starting {engine_display}')
        
        self._run_started_utc = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        start_t = time.time()

        self._configure_reproducibility()

        self.load()
        self.calc()
        self.save()

        self._total_runtime_s = float(time.time() - start_t)
        self._run_finished_utc = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        self._write_run_manifest()

        return 

def run(cfg_dct):
    from dbsipy.core.runner import run as _run

    return _run(cfg_dct)


def run_from_ui() -> Type[configparser.ConfigParser]:
    from dbsipy.core.runner import run_from_ui as _run_from_ui

    return _run_from_ui()
