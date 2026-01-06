CLI
===

DBSIpy installs the ``DBSIpy`` console script.

Help
----

.. code-block:: bash

   DBSIpy --help

Run
---

.. code-block:: bash

   DBSIpy run --help
   DBSIpy run --cfg_path PATH/TO/CONFIGURATION.ini

Output modes
------------

DBSIpy supports four console output modes:

- ``quiet``: minimal milestone updates (no progress bars)
- ``standard``: milestone updates + key config summary + progress bars
- ``verbose``: expanded user-facing details (the ``log`` file always looks like this)
- ``debug``: verbose + developer/debug diagnostics (written to ``debug_log``)

Select a mode via config (``[DEBUG] output_mode = ...``) or override via CLI:

.. code-block:: bash

   DBSIpy run --cfg_path PATH/TO/CONFIGURATION.ini --output_mode quiet
   DBSIpy benchmark --cfg_path PATH/TO/CONFIGURATION.ini --output_mode debug

Benchmark
---------

.. code-block:: bash

   DBSIpy benchmark --help

The benchmark command writes a consolidated JSON summary including per-run status, timings,
process-scoped CPU/RAM/I/O deltas, CUDA peak memory (when applicable), and the effective
configuration used for each run.

On shared servers, reporting is scoped to the DBSIpy process. You can disable peak RSS
sampling with:

.. code-block:: bash

   DBSIpy benchmark --no_resource_monitor

To change the sampling rate:

.. code-block:: bash

   DBSIpy benchmark --resource_sample_interval_s 0.5
