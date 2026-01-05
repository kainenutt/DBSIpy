from __future__ import annotations

import configparser
import logging
import os
import sys
import time
from pathlib import Path
from typing import Type

from dbsipy.core.validation import ConfigurationError
from dbsipy.gui.tk_dialogs import ask_dropdown_choice
from dbsipy.configs.paths import resolve_basis_path, resolve_config_path


def run(cfg_dct):
    """Entrypoint used by CLI/UI to run a DBSIpy pipeline from a config path or GUI."""
    if cfg_dct['cfg_path']:
        input_cfg_file = configparser.ConfigParser()
        input_cfg_file.read(cfg_dct['cfg_path'])

        if not input_cfg_file.has_section('DEBUG'):
            input_cfg_file.add_section('DEBUG')
        input_cfg_file.set('DEBUG', 'cfg_source', cfg_dct['cfg_path'])
    else:
        input_cfg_file = run_from_ui()

    from dbsipy.core.fast_DBSI import DBSIpy

    start = time.time()
    DBSIpy(input_cfg_file).__call__()
    end = time.time()
    logging.info(f"Total Runtime: {round(end-start,4)} sec")


def run_from_ui() -> Type[configparser.ConfigParser]:
    input_cfg_file = configparser.ConfigParser()

    # Lazy import Tkinter so config/CLI runs work on headless systems.
    try:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        from tkinter.simpledialog import askstring, askfloat, askinteger
        from tkinter.messagebox import askyesno, showerror
    except Exception as e:
        raise RuntimeError(
            "GUI mode requires Tkinter, but it could not be imported. "
            "If you're on Linux, install Tk support (often via python3-tk)."
        ) from e

    root = Tk()
    root.withdraw()
    # Ensure Tk is fully initialized; without this, some systems (notably remote
    # Linux sessions) can fail to show subsequent dialogs after a messagebox.
    try:
        root.update()
    except Exception:
        pass

    gui_debug = os.environ.get('DBSIPY_GUI_DEBUG', '0') == '1'

    def _dbg(msg: str) -> None:
        if not gui_debug:
            return
        try:
            print(f"[DBSI GUI] {msg}")
            sys.stdout.flush()
        except Exception:
            pass

    try:
        root.update_idletasks()
    except Exception:
        pass

    _dbg('Selecting DWI file...')
    dwi_file_path = askopenfilename(
        title='Select DWI File',
        initialdir=os.getcwd(),
        filetypes=[('NIfTI Image', '*.nii*'), ('All Files', '*.*')],
        parent=root,
    )
    assert len(dwi_file_path) > 0, 'No DWI selected. Select an image file to proceed.'
    _dbg(f'DWI selected: {dwi_file_path}')

    roi_file_path = askopenfilename(
        title='Select ROI/Mask File (Optional)',
        initialdir=os.path.dirname(dwi_file_path),
        filetypes=[('NIfTI Image', '*.nii*'), ('All Files', '*.*')],
        parent=root,
    )
    _dbg('ROI/mask selection complete.')

    bval_file_path = askopenfilename(
        title='Select B-Value File',
        initialdir=os.path.dirname(dwi_file_path),
        filetypes=[('bval file', '*bval*'), ('Text Files', '*.txt'), ('All Files', '*.*')],
        parent=root,
    )

    assert len(bval_file_path) > 0, 'No b-value file selected. Select a b-value file to proceed.'
    _dbg(f'BVAL selected: {bval_file_path}')

    bvec_file_path = askopenfilename(
        title='Select B-Vector File',
        initialdir=os.path.dirname(dwi_file_path),
        filetypes=[('bvec file', '*bvec*'), ('Text Files', '*.txt'), ('All Files', '*.*')],
        parent=root,
    )

    assert len(bvec_file_path) > 0, 'No b-vector file selected. Select a b-vector file to proceed.'
    _dbg(f'BVEC selected: {bvec_file_path}')

    # Prefer shipped config templates inside the installed package.
    try:
        from dbsipy.configs import get_configs_dir

        config_dir = get_configs_dir()
    except Exception:
        config_dir = Path(__file__).resolve().parent

    base_configuration_file = askopenfilename(
        title='Select Base Configuration File',
        initialdir=(
            str(config_dir)
            if hasattr(config_dir, 'exists') and config_dir.exists()
            else os.path.dirname(os.path.realpath(__file__))
        ),
        filetypes=[('Config file', '*.ini*'), ('All Files', '*.*')],
        parent=root,
    )

    assert len(base_configuration_file) > 0, 'No configuration file selected. Select a configuration file to proceed.'
    _dbg(f'Base config selected: {base_configuration_file}')

    verbose_flag = askyesno(
        title='Verbose Logging for Debugging?',
        message='Would you like to output all logging parameters (primarily used for debugging)?',
        parent=root,
    )
    try:
        root.update()
    except Exception:
        pass
    _dbg(f'Verbose selected: {verbose_flag}')

    _dbg('Selecting analysis engine...')
    computational_engine = ask_dropdown_choice(
        root,
        title='Select Analysis Engine',
        prompt='Select the analysis engine:',
        options=['DBSI', 'DBSI-IA', 'DTI', 'NODDI'],
        initial='DBSI',
    )

    computational_engine = (computational_engine or '').strip().upper()
    # UI label -> internal engine key
    if computational_engine in {'DBSI-IA', 'DBSI_IA', 'DBSI IA'}:
        computational_engine = 'IA'
    _dbg(f'Engine selected: {computational_engine}')

    assert len(computational_engine) > 0, 'Please select a computational engine.'
    assert computational_engine in ['DBSI', 'IA', 'DTI', 'NODDI'], (
        f"Invalid engine '{computational_engine}'. Must be DBSI, IA, DTI, or NODDI."
    )

    output_map_set = ask_dropdown_choice(
        root,
        title='Output Maps',
        prompt=(
            'Select output map set:\n\n'
            '- default: default maps only\n'
            '- expanded: default + expanded maps'
        ),
        options=['default', 'expanded'],
        initial='default',
    )
    output_map_set = (output_map_set or 'default').strip().lower()
    _dbg(f'Output map set selected: {output_map_set}')

    signal_normalization = ask_dropdown_choice(
        root,
        title='Signal Normalization',
        prompt=(
            'Select signal normalization mode:\n\n'
            '- auto: engine default (recommended)\n'
            '- max: divide by per-voxel max\n'
            '- b0: divide by b0 volume\n'
            '- minb: divide by minimum-b volume\n'
            '- none: no normalization'
        ),
        options=['auto', 'max', 'b0', 'minb', 'none'],
        initial='auto',
    )
    signal_normalization = (signal_normalization or 'auto').strip().lower()
    _dbg(f'Signal normalization selected: {signal_normalization}')

    learnable_s0 = askyesno(
        title='Learnable S0?',
        message='Enable learnable voxelwise S0 scaling (where supported)?',
        parent=root,
    )
    _dbg(f'Learnable S0 selected: {learnable_s0}')

    if computational_engine == 'NODDI':
        edit_optimizer_params = False
        edit_noddi_params = askyesno(
            title='Change NODDI Parameters?',
            message='Would you like to change NODDI optimizer parameters from their default values?',
        )
        if edit_noddi_params:
            noddi_lr = askfloat(
                title='NODDI Learning Rate',
                prompt='Enter a learning rate for NODDI optimization.',
                initialvalue=0.001,
                minvalue=1e-6,
                maxvalue=1e-2,
            )
            noddi_epochs = askinteger(
                title='NODDI Epochs',
                prompt='Enter the number of epochs for NODDI optimization.',
                initialvalue=500,
                minvalue=10,
                maxvalue=2000,
            )
            noddi_d_ic = askfloat(
                title='NODDI Intra-cellular Diffusivity',
                prompt='Enter intra-cellular diffusivity (mm²/s).',
                initialvalue=1.7e-3,
                minvalue=1e-3,
                maxvalue=3e-3,
            )
            noddi_d_iso = askfloat(
                title='NODDI Isotropic Diffusivity',
                prompt='Enter isotropic (free water) diffusivity (mm²/s).',
                initialvalue=3.0e-3,
                minvalue=2e-3,
                maxvalue=4e-3,
            )
            noddi_use_tortuosity = askyesno(
                title='Use Tortuosity Constraint?',
                message='Apply tortuosity constraint to extra-cellular diffusion?',
            )
    elif computational_engine == 'DTI':
        edit_optimizer_params = False
        edit_dti_params = askyesno(
            title='Change DTI Parameters?',
            message='Would you like to change DTI fitting parameters from their default values?',
            parent=root,
        )
        if edit_dti_params:
            dti_fit_method = ask_dropdown_choice(
                root,
                title='DTI Fit Method',
                prompt='Select fit method for DTI:',
                options=['WLS', 'OLS', 'ADAM'],
                initial='WLS',
            )
            dti_fit_method = (dti_fit_method or 'WLS').strip().upper()
            if dti_fit_method == 'ADAM':
                dti_lr = askfloat(
                    title='DTI ADAM Learning Rate',
                    prompt='Enter learning rate for DTI ADAM optimization.',
                    initialvalue=0.01,
                    minvalue=1e-6,
                    maxvalue=1.0,
                )
                dti_epochs = askinteger(
                    title='DTI ADAM Epochs',
                    prompt='Enter number of epochs for DTI ADAM optimization.',
                    initialvalue=200,
                    minvalue=10,
                    maxvalue=10000,
                )
            else:
                dti_lr = 0.01
                dti_epochs = 200
        else:
            dti_fit_method = 'WLS'
            dti_lr = 0.01
            dti_epochs = 200
    else:
        edit_optimizer_params = askyesno(
            title='Change Optimizer Parameters?',
            message='Would you like to change the optimizer parameters from their default values?',
        )

    if edit_optimizer_params:
        step_1_LR = askfloat(
            title='Step 1 Learning Rate',
            prompt='Enter a learning rate for step 1.',
            initialvalue=0.001,
            minvalue=1e-6,
            maxvalue=1e-2,
        )
        step_1_epochs = askinteger(
            title='Step 1 Epochs',
            prompt='Enter the number of epochs for step 1.',
            initialvalue=100,
            minvalue=10,
            maxvalue=1000,
        )
        step_1_loss_fn = askstring(
            title='Step 1 Loss Function',
            prompt='Enter "mse", "ridge", or "lasso" to select the loss function for step 1.',
            initialvalue='mse',
        )

        step_2_LR = askfloat(
            title='Step 2 Learning Rate',
            prompt='Enter a learning rate for step 2.',
            initialvalue=0.001,
            minvalue=1e-7,
            maxvalue=1e-3,
        )
        step_2_epochs = askinteger(
            title='Step 2 Epochs',
            prompt='Enter the number of epochs for step 2.',
            initialvalue=250,
            minvalue=100,
            maxvalue=5000,
        )
        step_2_loss_fn = askstring(
            title='Step 2 Loss Function',
            prompt='Enter "mse", "ridge", or "lasso" to select the loss function for step 2.',
            initialvalue='mse',
        )

        assert step_1_LR > 0, (
            'Error. No learning rate selected for step 1. Enter a number between 0.000001 and 0.01 to proceed.'
        )
        assert step_1_epochs > 0, (
            'Error. Number of epochs not selected for step 1. Enter an integer between 10 and 1000 to proceed.'
        )
        assert len(step_1_loss_fn) > 0, (
            'Error. No loss function selected for step 1. Enter mse, ridge, or lasso to proceed.'
        )
        assert step_2_LR > 0, (
            'Error. No learning rate selected for step 2. Enter a number between 0.0000001 and 0.001 to proceed.'
        )
        assert step_2_epochs > 0, (
            'Error. Number of epochs not selected for step 2. Enter an integer between 100 and 5000 to proceed.'
        )
        assert len(step_2_loss_fn) > 0, (
            'Error. No loss function selected for step 2. Enter mse, ridge, or lasso to proceed.'
        )

    # Global DBSI/IA parameters
    edit_global_params = False
    if computational_engine in ['DBSI', 'IA']:
        edit_global_params = askyesno(
            title='Change Global Parameters?',
            message='Would you like to change the global fitting parameters from their default values?',
        )
        if edit_global_params:
            num_fibers = askinteger(
                title='Number of Fibers to Model',
                prompt='Enter the desired number of fibers to allow in the model.',
                initialvalue=1,
                minvalue=1,
                maxvalue=3,
            )
            fiber_thresh = askfloat(
                title='Anisotropic Signal Threshold',
                prompt='Enter the desired threshold for anisotropic signals.',
                initialvalue=0.01,
                minvalue=0.001,
                maxvalue=0.2,
            )

    # DTI bval cutoff is used by all engines
    edit_dti_bval = askyesno(
        title='Change DTI B-Value Cutoff?',
        message='Would you like to change the maximum b-value used for DTI fitting?',
    )
    if edit_dti_bval:
        dti_bval_cut = askinteger(
            title='Max B-Value for DTI',
            prompt=(
                'Enter the maximum b-value to be used for DTI fitting (in units of s/mm^2). '
                'Default 1500 is a standard low-b DTI cutoff; 15000 uses all b-values (nonstandard for DTI).'
            ),
            initialvalue=1500,
            minvalue=1000,
            maxvalue=20000,
        )
    else:
        dti_bval_cut = 1500

    # Step 1 parameters (DBSI/IA only)
    edit_step1_params = False
    if computational_engine in ['DBSI', 'IA']:
        edit_step1_params = askyesno(
            title='Change Step 1 Parameters?',
            message='Would you like to change the step 1 fitting parameters from their default values?',
        )
        if edit_step1_params:
            ang_thresh = askinteger(
                title='Angle Threshold for Anisotropic Grouping',
                prompt='Enter the desired angular threshold (in degrees) to be used for anisotropic signal grouping.',
                initialvalue=30,
                minvalue=15,
                maxvalue=90,
            )
            step1_axi = askfloat(
                title='Step 1 Axial Diffusivity',
                prompt='Enter the desired axial diffusivty value for step 1 (in units of mm^2/s).',
                initialvalue=1.5e-3,
                minvalue=1e-3,
                maxvalue=3e-3,
            )
            step1_rad = askfloat(
                title='Step 1 Radial Diffusivity',
                prompt='Enter the desired radial diffusivty value for step 1 (in units of mm^2/s).',
                initialvalue=2e-4,
                minvalue=0.0,
                maxvalue=1e-3,
            )

    edit_ia_thresh = False
    if computational_engine == 'IA':
        edit_ia_thresh = askyesno(
            title='Change Intra-Axonal RD Threshold?',
            message='Would you like to change the radial diffusivity threshold for the intra-axonal signal from its default value?',
        )
        if edit_ia_thresh:
            ia_thresh = askfloat(
                title='Intra-Axonal RD Threshold',
                prompt='Enter the desired RD threshold (in units of mm^2/s) to be used for intra-axonal signal partitioning.',
                initialvalue=2e-4,
                minvalue=1e-6,
                maxvalue=1e-3,
            )

    # Isotropic thresholds (DBSI/IA only)
    edit_iso_cuts = False
    if computational_engine in ['DBSI', 'IA']:
        edit_iso_cuts = askyesno(
            title='Change Isotropic Thresholds?',
            message='Would you like to change the isotropic diffusivity thresholds from their default values?',
        )
    if edit_iso_cuts:
        use_highly_restricted = askyesno(
            title='Use Highly-Restricted Threshold?',
            message='Would you like to include a highly-restricted segment of the isotropic diffusion spectrum?',
        )
        if use_highly_restricted:
            hi_res_thresh = askfloat(
                title='Highly-Restricted Threshold',
                prompt='Enter the desired diffusivity threshold (in units of mm^2/s) for highly-restricted isotropic diffusion.',
                initialvalue=1e-4,
                minvalue=0,
                maxvalue=9e-4,
            )
        restrict_thresh = askfloat(
            title='Restricted Threshold',
            prompt='Enter the desired diffusivity threshold (in units of mm^2/s) for restricted isotropic diffusion.',
            initialvalue=3e-4,
            minvalue=0,
            maxvalue=9e-4,
        )
        water_thresh = askfloat(
            title='Free Water Threshold',
            prompt='Enter the desired diffusivity threshold (in units of mm^2/s) for free water isotropic diffusion.',
            initialvalue=3e-3,
            minvalue=1.5e-3,
            maxvalue=3.5e-3,
        )

    # Configure Configuration File
    base_configuration_file = resolve_config_path(base_configuration_file)
    input_cfg_file.read(base_configuration_file)

    if not input_cfg_file.has_section('DEBUG'):
        input_cfg_file.add_section('DEBUG')
    input_cfg_file.set('DEBUG', 'cfg_source', base_configuration_file)

    # Ensure required sections exist (in case base config is for different engine)
    required_sections = ['INPUT', 'DEBUG', 'GLOBAL']
    for section in required_sections:
        if not input_cfg_file.has_section(section):
            input_cfg_file.add_section(section)

    # Engine-specific sections
    if computational_engine in ['DBSI', 'IA']:
        if not input_cfg_file.has_section('OPTIMIZER'):
            input_cfg_file.add_section('OPTIMIZER')
        if not input_cfg_file.has_section('STEP_1'):
            input_cfg_file.add_section('STEP_1')
        if not input_cfg_file.has_section('STEP_2'):
            input_cfg_file.add_section('STEP_2')
        if not input_cfg_file.has_section('ISOTROPIC'):
            input_cfg_file.add_section('ISOTROPIC')
    elif computational_engine == 'DTI':
        if not input_cfg_file.has_section('DTI'):
            input_cfg_file.add_section('DTI')

    # Configure Input Data
    input_cfg_file.set('INPUT', 'dwi_file', dwi_file_path)
    input_cfg_file.set('INPUT', 'bval_file', bval_file_path)
    input_cfg_file.set('INPUT', 'bvec_file', bvec_file_path)
    if len(roi_file_path) > 0:
        input_cfg_file.set('INPUT', 'mask_file', roi_file_path)
    else:
        # Match pipeline behavior: missing mask -> auto-masking.
        input_cfg_file.set('INPUT', 'mask_file', 'auto')

    # Configure Optimizer Parameters
    if edit_optimizer_params:
        input_cfg_file.set('OPTIMIZER', 'step_1_lr', str(step_1_LR))
        input_cfg_file.set('OPTIMIZER', 'step_1_epochs', str(step_1_epochs))
        input_cfg_file.set('OPTIMIZER', 'step_1_loss_fn', str(step_1_loss_fn))

        input_cfg_file.set('OPTIMIZER', 'step_2_lr', str(step_2_LR))
        input_cfg_file.set('OPTIMIZER', 'step_2_epochs', str(step_2_epochs))
        input_cfg_file.set('OPTIMIZER', 'step_2_loss_fn', str(step_2_loss_fn))

    # Configure Debugging Parameter(s)
    input_cfg_file.set('DEBUG', 'verbose', str(verbose_flag))

    # Configure Global Fitting Parameters (DBSI/IA specific)
    if edit_global_params and computational_engine in ['DBSI', 'IA']:
        input_cfg_file.set('GLOBAL', 'max_group_number', str(num_fibers))
        input_cfg_file.set('GLOBAL', 'fiber_threshold', str(fiber_thresh))

    # Configure DTI bval cutoff (all engines)
    if edit_dti_bval:
        input_cfg_file.set('GLOBAL', 'dti_bval_cut', str(dti_bval_cut))

    input_cfg_file.set('GLOBAL', 'model_engine', str(computational_engine))
    input_cfg_file.set('GLOBAL', 'output_map_set', str(output_map_set))
    input_cfg_file.set('GLOBAL', 'signal_normalization', str(signal_normalization))
    input_cfg_file.set('GLOBAL', 'learnable_s0', str(bool(learnable_s0)))

    # Configure NODDI Parameters (only if NODDI engine selected)
    if computational_engine == 'NODDI':
        # Ensure NODDI section exists
        if not input_cfg_file.has_section('NODDI'):
            input_cfg_file.add_section('NODDI')

        if edit_noddi_params:
            input_cfg_file.set('NODDI', 'noddi_lr', str(noddi_lr))
            input_cfg_file.set('NODDI', 'noddi_epochs', str(noddi_epochs))
            input_cfg_file.set('NODDI', 'noddi_d_ic', str(noddi_d_ic))
            input_cfg_file.set('NODDI', 'noddi_d_iso', str(noddi_d_iso))
            input_cfg_file.set('NODDI', 'noddi_use_tortuosity', str(noddi_use_tortuosity))
        else:
            # Set defaults
            input_cfg_file.set('NODDI', 'noddi_lr', '0.001')
            input_cfg_file.set('NODDI', 'noddi_epochs', '500')
            input_cfg_file.set('NODDI', 'noddi_d_ic', '0.0017')
            input_cfg_file.set('NODDI', 'noddi_d_iso', '0.003')
            input_cfg_file.set('NODDI', 'noddi_use_tortuosity', 'True')

    # Configure DTI Parameters (only if DTI engine selected)
    if computational_engine == 'DTI':
        if not input_cfg_file.has_section('DTI'):
            input_cfg_file.add_section('DTI')
        input_cfg_file.set('DTI', 'dti_fit_method', str(dti_fit_method))
        input_cfg_file.set('DTI', 'dti_lr', str(dti_lr))
        input_cfg_file.set('DTI', 'dti_epochs', str(dti_epochs))

    # Configure Step 1 Fitting Parameters
    if edit_step1_params:
        input_cfg_file.set('STEP_1', 'angle_threshold', str(ang_thresh))
        input_cfg_file.set('STEP_1', 'step_1_axial', str(step1_axi))
        input_cfg_file.set('STEP_1', 'step_1_radial', str(step1_rad))

    # Configure Step 2 Fitting Parameters
    if edit_ia_thresh:
        input_cfg_file.set('STEP_2', 'intra_threshold', str(ia_thresh))

    # Configure Isotropic Segmentation Parameters
    if edit_iso_cuts:
        input_cfg_file.set('ISOTROPIC', 'restricted_threshold', str(restrict_thresh))
        input_cfg_file.set('ISOTROPIC', 'free_water_threshold', str(water_thresh))
        if use_highly_restricted:
            input_cfg_file.set('ISOTROPIC', 'highly_restricted_threshold', str(hi_res_thresh))

    # Preflight: fail fast if base config mismatches engine selection (missing required DBSI/IA keys).
    # This avoids later KeyError/ValueError crashes in configuration._setup_config.
    if computational_engine in {'DBSI', 'IA'}:
        required: list[tuple[str, str]] = [
            ('GLOBAL', 'max_group_number'),
            ('GLOBAL', 'fiber_threshold'),
            ('STEP_1', 'angle_threshold'),
            ('STEP_1', 'angle_basis'),
            ('STEP_1', 'iso_basis'),
            ('STEP_1', 'step_1_axial'),
            ('STEP_1', 'step_1_radial'),
            ('STEP_2', 'step_2_axials'),
            ('STEP_2', 'step_2_radials'),
            ('STEP_2', 'intra_threshold'),
            ('ISOTROPIC', 'restricted_threshold'),
            ('ISOTROPIC', 'free_water_threshold'),
        ]
        missing: list[str] = []
        for sec, opt in required:
            if not input_cfg_file.has_option(sec, opt):
                missing.append(f'[{sec}] {opt}')
                continue
            if str(input_cfg_file.get(sec, opt, fallback='')).strip() == '':
                missing.append(f'[{sec}] {opt} (empty)')

        missing_files: list[str] = []
        cfg_source = None
        try:
            if input_cfg_file.has_section('DEBUG') and input_cfg_file.has_option('DEBUG', 'cfg_source'):
                cfg_source = str(input_cfg_file.get('DEBUG', 'cfg_source')).strip() or None
        except Exception:
            cfg_source = None

        for sec, opt in [
            ('STEP_1', 'angle_basis'),
            ('STEP_1', 'iso_basis'),
            ('STEP_2', 'step_2_axials'),
            ('STEP_2', 'step_2_radials'),
        ]:
            raw = str(input_cfg_file.get(sec, opt, fallback='')).strip()
            p = resolve_basis_path(raw, cfg_source=cfg_source)
            if p and p != raw:
                input_cfg_file.set(sec, opt, p)

            if not p or not os.path.exists(p):
                missing_files.append(f'[{sec}] {opt} -> {p or "<empty>"}')

        if missing or missing_files:
            msg = (
                'Selected engine requires DBSI/IA configuration fields that are missing from the chosen base .ini.\n\n'
                'Missing options:\n'
                + ("\n".join(f'- {m}' for m in missing) if missing else '(none)')
                + '\n\nMissing/invalid basis paths:\n'
                + ("\n".join(f'- {m}' for m in missing_files) if missing_files else '(none)')
                + '\n\nAction: choose an IA/DBSI base config (e.g., one of the Human/Mouse configs) or fill these fields in the .ini.'
            )
            try:
                showerror(title='Invalid Base Configuration', message=msg, parent=root)
            except Exception:
                pass
            raise ConfigurationError(msg)

    try:
        root.destroy()
    except Exception:
        pass

    return input_cfg_file
