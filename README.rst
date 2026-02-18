.. image :: https://github.com/kainenutt/DBSIpy/blob/main/figures/Logo.png
 :alt: DBSIPy logo


This library, ``DBSIPy``, implements Diffusion Basis Spectrum Imaging with a PyTorch backend to take advantage of modern machine learning optimization algorithms and hardware acceleration. If you utlize DBSIpy in your work, please cite the associated publications:

Utt, K.L., Blum, J.S., Rim, D. and Song, S.K., 2026. Accelerated Diffusion Basis Spectrum Imaging With Tensor Computations. Human Brain Mapping, 47(2), p.e70460. https://doi.org/10.1002/hbm.70460.

Installation and Usage
----------------------

Compatibility
~~~~~~~~~~~~~
``DBSIPy`` supports Python 3.8+.

GPU acceleration is optional. If you install a CUDA-enabled build of PyTorch and your system has a compatible NVIDIA GPU + driver, DBSIpy will use it; otherwise it will run on CPU.

Core runtime dependencies are managed by pip (see ``requirements.txt`` / ``setup.py``):

- **PyTorch:** https://pytorch.org/
- **NumPy:** https://numpy.org/
- **DIPY:** https://dipy.org/
- **SciPy:** https://scipy.org/
- **Nibabel:** https://nipy.org/nibabel/
- **Pandas:** https://pandas.pydata.org/
- **Psutil:** https://github.com/giampaolo/psutil
- **Joblib:** https://joblib.readthedocs.io/en/latest/index.html

Installation
~~~~~~~~~~~~

DBSIpy is currently intended to be installed from source (this repository).

1) Install PyTorch

PyTorch wheels vary by OS/CUDA. Follow the official selector for your platform:
https://pytorch.org/get-started/locally/

2) Install DBSIpy

From the repository root:

.. code-block:: bash

     python -m pip install -U pip
     python -m pip install -e [PATH_TO_REPOSITORY]

For development (linters/tests tooling), also install:

.. code-block:: bash

     python -m pip install -r requirements-dev.txt

Testing
~~~~~~~

The default test suite is *mechanistic/operational* and is safe to run frequently:

.. code-block:: bash

     python -m pytest

DBSIpy also contains opt-in *physical accuracy* tests using small simulated phantoms.
These are skipped by default and only run when explicitly requested:

.. code-block:: bash

     python -m pytest -m accuracy

Running DBSIpy
~~~~~~~~~~~~~~

The installed command-line entrypoint is ``DBSIpy``.

.. code-block:: bash

     DBSIpy --help
     DBSIpy run --help
     DBSIpy benchmark --help

Run from a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run DBSIpy using a pre-existing configuration file:

.. code-block:: bash

     DBSIpy run --cfg_path PATH/TO/CONFIGURATION.ini

GUI (interactive) mode
~~~~~~~~~~~~~~~~~~~~~~
If you omit the ``--cfg_path`` argument, DBSIpy will attempt to launch an interactive GUI flow (using Tkinter) to guide you through configuration and execution.

.. code-block:: bash

     DBSIpy run

Configuration templates
~~~~~~~~~~~~~~~~~~~~~~~

This repository ships example configuration files under ``dbsipy/configs/``:

- ``dbsipy/configs/Template_All_Engines_Minimal.ini`` (minimal starting point)
- ``dbsipy/configs/DBSI_Configs/*.ini`` (DBSI/DBSI-IA engine configs)

Basis-set CSVs referenced by DBSI/DBSI-IA configs are shipped under
``dbsipy/configs/DBSI_Configs/BasisSets/`` and are resolved at runtime.

Benchmark mode
~~~~~~~~~~~~~~

To run a benchmark that executes all supported engines on the same dataset and writes a consolidated summary JSON:

.. code-block:: bash

     DBSIpy benchmark \
       --cfg_path PATH/TO/CONFIGURATION.ini \
       --dwi_file PATH/TO/DWI.nii.gz \
       --bval_file PATH/TO/bvals \
       --bvec_file PATH/TO/bvecs \
       --mask_file auto

The benchmark summary JSON includes per-run engine status, wall time, per-process CPU/RAM/I/O deltas, CUDA peak memory
(when applicable), and a snapshot of the effective configuration used for each run.

On shared servers, resource reporting is scoped to the current DBSIpy process (not whole-node utilization). You can
disable peak RSS sampling with:

.. code-block:: bash

           DBSIpy benchmark --no_resource_monitor ...

Running from a git checkout (no install)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer not to install the package, you can run the same CLI from the repository root:

.. code-block:: bash

     python master_cli.py --help
     python master_cli.py run --cfg_path PATH/TO/CONFIGURATION.ini
