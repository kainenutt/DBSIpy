
.. image :: https://github.com/kainenutt/DBSIpy/blob/45d6dc8f79e8591d72877bbb19881d729fd84c22/figures/Logo.png
 :alt: DBSIPy logo


This library, ``DBSIPy``, implements Diffusion Basis Spectrum Imaging with a PyTorch backend to take advantage of modern machine learning optimization algorithms and hardware acceleration.

Installation and Usage 
----------------------

Compatibility
~~~~~~~~~~~~~~~~~~~~~
``DBSIPy`` is compatible with Python 3 or later, and requires a CUDA device with a compute capability of 3 or higher. We also require the following packages as dependencies. Installation
may need to be undertaken manually until we put ``DBSIPy`` on PyPI, which we're aiming to do soon.

- **PyTorch:** https://pytorch.org/
- **NumPy:** https://numpy.org/
- **DIPY:** https://dipy.org/
- **SciPy:** https://scipy.org/
- **Nibabel:** https://nipy.org/nibabel/
- **Pandas:** https://pandas.pydata.org/
- **Psutil:** https://github.com/giampaolo/psutil
- **Joblib:** https://joblib.readthedocs.io/en/latest/index.html

Manual installation
~~~~~~~~~~~~~~~~~~~

First, we need to install PyTorch and check that it is installed correctly by running the following commands:

.. code-block:: bash
   
   [USER]$ python3 

Now, type the following commands. If the installation is correct, the output should look something like this:

.. code-block:: bash

   >>> import torch 
   >>> torch.rand(1, 1, device = 'cuda')
   tensor([[0.7586]], device = 'cuda:0')

Exit the Python session and install the remaining dependencies using commands of the following form:

.. code-block:: bash
 
     [USER]$ python -m pip install --user DEPENDENCY 

Then, clone this repository and run "setup.py" to configure the code.

.. code-block:: bash
     
     [USER]$ git clone https://github.com/kainenutt/DBSIpy/
     [USER]$ python PATH_TO_DBSIPY_ROOT_DIRECTORY/setup.py 

Running DBSIpy
~~~~~~~~~~~~~~~~~~~~~~

This is a beta build of ``DBSIpy``. Note that the interface may change with future updates. 

To run DBSIpy in ``interactive mode``, use the following command:

.. code-block:: bash

     [USER]$ python PATH_TO_DBSIPY_ROOT_DIRECTORY/master_cli.py run


To run DBSIpy from a pre-determined configuration file, use the following command:

.. code-block:: bash

     [USER]$ python PATH_TO_DBSIpy_ROOT_DIRECTORY/master_cli.py run --cfg_path [PATH/TO/CONFIGURATION/FILE.ini]









