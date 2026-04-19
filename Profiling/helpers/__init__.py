"""
Vendored copies of eNest and nest_vnn (DrugCellNN) modules used only by profile_networks.py.

Source snapshots (for provenance):
- eNest: nest_vnn/eNest/eNest.py
- drugcell_nn, training_data_wrapper, util: vendored from nest_vnn ``src/`` (historical snapshot)
- nest_data_paths: resolves ``<repo>/Data`` or ``FNEST_ONTOLOGY_DIR`` for ontology files

Imports in this package use relative paths so profiling does not depend on ArchitecturePerformanceExperiments/ or external nest_vnn paths.
"""
