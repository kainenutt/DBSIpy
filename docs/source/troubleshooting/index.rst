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
