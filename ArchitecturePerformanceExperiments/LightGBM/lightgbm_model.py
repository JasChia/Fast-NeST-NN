"""LightGBM booster construction for the Fast-NeST-NN architecture comparison.

Mirrors the role of `fc_nn.py` / `fnest_nn.py` in the neural methods: this module
holds only the model definition (the parameter block and the search space), while
`lightgbm_hparam_tuner.py` drives Optuna and the train/val/test protocol.
"""
from __future__ import annotations

from typing import Any, Dict

import optuna

# Fixed across all trials. `deterministic` + `force_row_wise` are required for
# run-to-run reproducibility: without them LightGBM's histogram construction
# depends on thread count.
FIXED_PARAMS: Dict[str, Any] = {
    "objective": "regression",
    "metric": "l2",
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
}


def suggest_params(trial: optuna.Trial, seed: int, num_threads: int) -> Dict[str, Any]:
    """Sample one hyperparameter configuration."""
    params = dict(FIXED_PARAMS)
    params.update(
        {
            "seed": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "num_threads": num_threads,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 255, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 5, 8, 12]),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
    )
    return params
