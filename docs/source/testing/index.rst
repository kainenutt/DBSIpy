Testing
=======

DBSIpy uses pytest.

Default tests
-------------

The default suite is mechanistic/operational and runs by default:

.. code-block:: bash

   python -m pytest

Physical accuracy tests (opt-in)
--------------------------------

There is an opt-in set of physical-accuracy tests driven by small simulated phantoms.
These are marked with the ``accuracy`` marker and are skipped by default.

Run them explicitly with:

.. code-block:: bash

   python -m pytest -m accuracy
