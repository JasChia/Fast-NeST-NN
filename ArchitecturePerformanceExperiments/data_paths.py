"""
Shared path helpers for all `ArchitecturePerformanceExperiments/<Model>/` scripts.

Layout (default, self-contained clone):
  <repo>/Data/red_ontology.txt, red_gene2ind.txt, …
  <repo>/Data/nest_shuffle_data/CombatLog2TPM/Drug{N}/D{N}_CL/...

Override with env ``FAST_NEST_DATA_ROOT`` pointing at the directory that contains
``nest_shuffle_data/`` and ontology files (same layout as above relative to that root).
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root_from(file: str) -> Path:
    """Repo root (Fast-NeST-NN) from ``__file__`` in ``ArchitecturePerformanceExperiments/<Model>/*.py``."""
    return Path(file).resolve().parents[2]


def data_root_from(file: str) -> Path:
    """Data directory: ``$FAST_NEST_DATA_ROOT`` or ``<repo>/Data``."""
    root = repo_root_from(file)
    env = os.environ.get("FAST_NEST_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return root / "Data"


def drug_cl_dir(file: str, drug: int) -> Path:
    """Per-drug cell-line bundle: .../CombatLog2TPM/Drug{N}/D{N}_CL."""
    return (
        data_root_from(file)
        / "nest_shuffle_data"
        / "CombatLog2TPM"
        / f"Drug{drug}"
        / f"D{drug}_CL"
    )


def nest_combat_root(file: str) -> Path:
    """``.../nest_shuffle_data/CombatLog2TPM`` under data root."""
    return data_root_from(file) / "nest_shuffle_data" / "CombatLog2TPM"
