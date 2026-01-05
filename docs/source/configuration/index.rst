Configuration
=============

DBSIpy runs are driven by INI configuration files.

Templates
---------

- Minimal template: ``dbsipy/configs/Template_All_Engines_Minimal.ini``
- DBSI configs: ``dbsipy/configs/DBSI_Configs/*.ini``

Basis sets
----------

DBSI/DBSI-IA configs reference basis-set CSVs. These are shipped under:

- ``dbsipy/configs/DBSI_Configs/BasisSets/``

In configs, basis paths can be written as:

- ``BasisSets/<dataset>/<file>.csv``

and are resolved at runtime.
