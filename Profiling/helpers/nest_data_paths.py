"""
Resolve the directory containing NeST ontology inputs (red_ontology.txt, red_gene2ind.txt).

Resolution order:
1. Environment variable ``FNEST_ONTOLOGY_DIR`` (preferred) or ``NEST_VNN_DATA_DIR``
2. ``<repository-root>/Data`` where repository root is the parent of the ``Profiling`` folder

This keeps profiling portable after cloning the repo when ``Data/`` is present at the root.
"""

import os

_ONTOLOGY_BASENAME = "red_ontology.txt"
_GENE2ID_BASENAME = "red_gene2ind.txt"


def resolve_nest_data_dir():
    """Return absolute path to the folder that contains ontology and gene-ID files."""
    for key in ("FNEST_ONTOLOGY_DIR", "NEST_VNN_DATA_DIR"):
        v = os.environ.get(key, "").strip()
        if v:
            return os.path.abspath(os.path.expanduser(v))
    helpers_dir = os.path.dirname(os.path.abspath(__file__))
    profiling_dir = os.path.dirname(helpers_dir)
    repo_root = os.path.abspath(os.path.join(profiling_dir, ".."))
    return os.path.join(repo_root, "Data")


def nest_data_file_paths(data_dir=None):
    """Return (ontology_path, gene2id_path)."""
    d = data_dir if data_dir is not None else resolve_nest_data_dir()
    d = os.path.abspath(os.path.expanduser(d))
    return (
        os.path.join(d, _ONTOLOGY_BASENAME),
        os.path.join(d, _GENE2ID_BASENAME),
    )


def ensure_nest_data_files(data_dir=None):
    """
    Raise FileNotFoundError with a clear message if required files are missing.
    Call after setting FNEST_ONTOLOGY_DIR (e.g. from CLI --data-dir).
    """
    d = data_dir if data_dir is not None else resolve_nest_data_dir()
    ont, g2i = nest_data_file_paths(d)
    missing = [p for p in (ont, g2i) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "NeST profiling requires {!r} and {!r} under {!r}. Missing: {}".format(
                _ONTOLOGY_BASENAME,
                _GENE2ID_BASENAME,
                d,
                missing,
            )
        )
