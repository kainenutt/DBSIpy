Troubleshooting
===============

CLI not found
-------------

If ``DBSIpy`` is not found after installation:

- Ensure you are in the same environment where you installed DBSIpy.
- On some shells (notably csh/tcsh), run ``rehash`` after installing.

Import issues on Windows
------------------------

Prefer running tests with:

.. code-block:: bash

   python -m pytest

This ensures pytest uses the same interpreter you installed DBSIpy into.

Shared servers / multi-GPU
--------------------------

When running on shared servers (or multi-GPU workstations):

- Prefer selecting GPUs via ``CUDA_VISIBLE_DEVICES`` (or your scheduler). DBSIpy follows PyTorch's
   current CUDA device; avoid assumptions that GPU ``0`` is always the intended device.
- Memory availability can be constrained by scheduler/cgroup limits. If you see OOM behavior despite
   system-wide free RAM, verify your job's memory allocation.
