<<<<<<< HEAD

.. image :: https://github.com/kainenutt/DBSIpy/blob/03f752a4d37811baa170d6c28ca42db1f8aff01a/figures/Logo.png
 :alt: DBSIPy logo


This library, ``DBSIPy``, implements Diffusion Basis Spectrum Imaging with a PyTorch backend to take advantage of modern machine learning optimization algorithms and hardware acceleration.

Installation and Usage 
----------------------

Compatibility
~~~~~~~~~~~~~~~~~~~~~
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
~~~~~~~~~~~~~~~~~~~

DBSIpy is currently intended to be installed from source (this repository).

1) Install PyTorch

PyTorch wheels vary by OS/CUDA. Follow the official selector for your platform:
https://pytorch.org/get-started/locally/

2) Install DBSIpy

From the repository root:

.. code-block:: bash

     python -m pip install -U pip
     python -m pip install -e [PATH_TO_REPOSITORY]
.. image :: https://github.com/kainenutt/DBSIpy/blob/main/figures/Logo.png
 :alt: DBSIPy logo


This library, ``DBSIPy``, implements Diffusion Basis Spectrum Imaging with a PyTorch backend to take advantage of modern machine learning optimization algorithms and hardware acceleration.

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

Running DBSIpy
~~~~~~~~~~~~~~

The installed command-line entrypoint is ``DBSI``.

.. code-block:: bash

     DBSI --help
     DBSI run --help
     DBSI benchmark --help

Run from a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run DBSIpy using a pre-existing configuration file:

.. code-block:: bash

     DBSI run --cfg_path PATH/TO/CONFIGURATION.ini

GUI (interactive) mode
~~~~~~~~~~~~~~~~~~~~~~

If you omit ``--cfg_path``, DBSIpy launches a GUI flow (file dialogs) to select inputs and a base configuration template. This requires Tkinter support in your Python installation.

.. code-block:: bash

     DBSI run

Configuration templates
~~~~~~~~~~~~~~~~~~~~~~~

This repository ships example configuration files under ``dbsipy/configs/*.ini``. Basis-set CSV paths referenced by those configs are also shipped under ``dbsipy/configs/BasisSets/`` and are resolved at runtime.

If you want a minimal starting point, see ``dbsipy/configs/Template_All_Engines_Minimal.ini``.

Benchmark mode
~~~~~~~~~~~~~~

To run a benchmark that executes all supported engines on the same dataset and writes a consolidated summary JSON:

.. code-block:: bash

     DBSI benchmark \
       --cfg_path PATH/TO/CONFIGURATION.ini \
       --dwi_file PATH/TO/DWI.nii.gz \
       --bval_file PATH/TO/bvals \
       --bvec_file PATH/TO/bvecs \
       --mask_file auto

Running from a git checkout (no install)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer not to install the package, you can run the same CLI from the repository root:

.. code-block:: bash

     python master_cli.py --help
     python master_cli.py run --cfg_path PATH/TO/CONFIGURATION.ini
.. code-block:: bash
