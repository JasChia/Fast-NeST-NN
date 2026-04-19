#!/usr/bin/env python3
"""Regenerate ArchitecturePerformanceExperiments/*/jobs/*_jobs.txt and *_advanced_jobs.json.

Transforms:
  - ANL_Drug_CData absolute paths -> ../../Data/ (repo-relative from each method folder)
  - Optional old_script -> new_script (e.g. eNest_hparam_tuner -> fnest_nn_hparam_tuner)
  - Drops commands for drug IDs 57 and 201 (paths and -drug)
  - Renumbers # Job N comments; rebuilds header stats from surviving lines
  - JSON: same filters/transforms on each job's "command"; updates drugs/total_jobs
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCHEDULER = Path(
    "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/jchia_l05_7-31-25/scheduler"
)
APE = Path(__file__).resolve().parent.parent / "ArchitecturePerformanceExperiments"
PREFIX = "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData/"
DROP_DRUGS = frozenset({57, 201})

# (relative_out, relative_scheduler_in, (old_script, new_script) or None for identity)
JOBSPECS: List[Tuple[str, str, Tuple[str, str] | None]] = [
    (
        "fNeST-NN/jobs/fnest_nn_jobs.txt",
        "eNest_linear_fair/jobs/eNest_linear_fair_jobs.txt",
        ("eNest_hparam_tuner.py", "fnest_nn_hparam_tuner.py"),
    ),
    (
        "Dense-fNeST/jobs/dense_fnest_jobs.txt",
        "fc_nn_DI_Layer_Pred/jobs/fc_nn_di_layer_pred_jobs.txt",
        ("fc_nn_di_layer_pred_hparam_tuner.py", "dense_fnest_hparam_tuner.py"),
    ),
    ("FCNN/jobs/fc_nn_jobs.txt", "fc_nn/jobs/fc_nn_jobs.txt", None),
    ("RSNN/jobs/r_sparse_nn_jobs.txt", "r_sparse_nn/jobs/r_sparse_nn_jobs.txt", None),
    ("NeST-VNN/jobs/nest_vnn_jobs.txt", "og_nest_vnn/jobs/nest_vnn_jobs.txt", None),
    (
        "GP-NN/jobs/global_prune_nn_warmup_jobs.txt",
        "global_prune_nn_warmup/jobs/global_prune_nn_warmup_jobs.txt",
        None,
    ),
    ("LP-NN/jobs/layer_prune_nn_jobs.txt", "layer_prune_nn/jobs/layer_prune_nn_jobs.txt", None),
    (
        "UGP-NN/jobs/relaxed_global_prune_nn_warmup_jobs.txt",
        "relaxed_global_prune_nn_warmup/jobs/relaxed_global_prune_nn_warmup_jobs.txt",
        None,
    ),
    (
        "RP-fNeST/jobs/uniform_random_do_di_snn_jobs.txt",
        "uniform_random_DO_DI/jobs/uniform_random_do_di_snn_jobs.txt",
        None,
    ),
    (
        "ERK_SNN/jobs/ERK_SNN_jobs.txt",
        "sparse_wd/ERK_SNN/jobs/ERK_SNN_jobs.txt",
        None,
    ),
]

JSONSPECS: List[Tuple[str, str, Tuple[str, str] | None]] = [
    (
        "fNeST-NN/jobs/fnest_nn_advanced_jobs.json",
        "eNest_linear_fair/jobs/eNest_linear_fair_advanced_jobs.json",
        ("eNest_hparam_tuner.py", "fnest_nn_hparam_tuner.py"),
    ),
    (
        "Dense-fNeST/jobs/dense_fnest_advanced_jobs.json",
        "fc_nn_DI_Layer_Pred/jobs/fc_nn_di_layer_pred_advanced_jobs.json",
        ("fc_nn_di_layer_pred_hparam_tuner.py", "dense_fnest_hparam_tuner.py"),
    ),
    ("FCNN/jobs/fc_nn_advanced_jobs.json", "fc_nn/jobs/fc_nn_advanced_jobs.json", None),
    ("RSNN/jobs/r_sparse_nn_advanced_jobs.json", "r_sparse_nn/jobs/r_sparse_nn_advanced_jobs.json", None),
    ("NeST-VNN/jobs/nest_vnn_advanced_jobs.json", "og_nest_vnn/jobs/nest_vnn_advanced_jobs.json", None),
    (
        "GP-NN/jobs/global_prune_nn_warmup_advanced_jobs.json",
        "global_prune_nn_warmup/jobs/global_prune_nn_warmup_advanced_jobs.json",
        None,
    ),
    (
        "LP-NN/jobs/layer_prune_nn_advanced_jobs.json",
        "layer_prune_nn/jobs/layer_prune_nn_advanced_jobs.json",
        None,
    ),
    (
        "UGP-NN/jobs/relaxed_global_prune_nn_warmup_advanced_jobs.json",
        "relaxed_global_prune_nn_warmup/jobs/relaxed_global_prune_nn_warmup_advanced_jobs.json",
        None,
    ),
    (
        "RP-fNeST/jobs/uniform_random_do_di_snn_advanced_jobs.json",
        "uniform_random_DO_DI/jobs/uniform_random_do_di_snn_advanced_jobs.json",
        None,
    ),
    (
        "ERK_SNN/jobs/ERK_SNN_advanced_jobs.json",
        "sparse_wd/ERK_SNN/jobs/ERK_SNN_advanced_jobs.json",
        None,
    ),
]

DRUG_RE = re.compile(r"-drug\s+(\d+)\b")


def drug_id_from_cmd(line: str) -> int | None:
    m = DRUG_RE.search(line)
    return int(m.group(1)) if m else None


def transform_line(line: str, script_pair: Tuple[str, str] | None) -> str:
    s = line.replace(PREFIX, "../../Data/")
    if script_pair:
        old, new = script_pair
        s = s.replace(old, new)
    return s


def process_file(src: Path, dst: Path, script_pair: Tuple[str, str] | None, title: str) -> None:
    text = src.read_text(encoding="utf-8", errors="replace").splitlines()
    out_cmds: List[str] = []
    for line in text:
        stripped = line.strip()
        if not stripped.startswith("python "):
            continue
        did = drug_id_from_cmd(stripped)
        if did is not None and did in DROP_DRUGS:
            continue
        out_cmds.append(transform_line(stripped, script_pair))

    drugs = sorted({drug_id_from_cmd(c) for c in out_cmds if drug_id_from_cmd(c) is not None})
    n = len(out_cmds)
    n_drugs = len(drugs)
    per = n // n_drugs if n_drugs else 0

    header = [
        title,
        f"# Generated automatically for {n_drugs} drugs",
        f"# Total jobs: {n}",
        f"# Drugs: {drugs}",
        f"# Experiments per drug: {per}",
        "# Hyperparameter tuning trials: 100",
        "# Seed: 49000",
        "#",
        "# Format: drug_experiment",
        "# Example: D5_0, D5_1, D5_2, etc.",
        "#",
        "",
    ]
    body: List[str] = []
    for i, cmd in enumerate(out_cmds, start=1):
        body.append(f"# Job {i}")
        body.append(cmd)
        body.append("")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(header + body).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {dst} ({n} jobs, drugs={drugs})")


def process_json(src: Path, dst: Path, script_pair: Tuple[str, str] | None, root_description: str | None) -> None:
    data: Dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))
    jobs_in = data.get("jobs") or []
    kept: List[Dict[str, Any]] = []
    for j in jobs_in:
        meta = j.get("metadata") or {}
        did = meta.get("drug_id")
        if did is not None and int(did) in DROP_DRUGS:
            continue
        cmd = j.get("command", "")
        j = dict(j)
        j["command"] = transform_line(cmd, script_pair)
        if "description" in j and isinstance(j["description"], str):
            j["description"] = j["description"].replace("eNest_linear_fair", "fNeST-NN").replace(
                "FC_NN_DI_Layer_Pred", "Dense_fNeST"
            )
        kept.append(j)

    drugs = sorted({int((x.get("metadata") or {}).get("drug_id")) for x in kept if (x.get("metadata") or {}).get("drug_id") is not None})
    data["jobs"] = kept
    data["total_jobs"] = len(kept)
    data["drugs"] = drugs
    if drugs:
        data["experiments_per_drug"] = len(kept) // len(drugs)
    if root_description:
        data["description"] = root_description
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {dst} ({len(kept)} jobs, drugs={drugs})")


def main() -> None:
    titles = {
        "fnest_nn_jobs.txt": "# fNeST-NN Hyperparameter Tuning Jobs",
        "dense_fnest_jobs.txt": "# Dense_fNeST Hyperparameter Tuning Jobs",
        "fc_nn_jobs.txt": "# FC_NN Hyperparameter Tuning Jobs",
        "r_sparse_nn_jobs.txt": "# Sparse_NN Hyperparameter Tuning Jobs",
        "nest_vnn_jobs.txt": "# NeST VNN Hyperparameter Tuning Jobs",
        "global_prune_nn_warmup_jobs.txt": "# GlobalPrunedFC_NN Hyperparameter Tuning Jobs (with warmup)",
        "layer_prune_nn_jobs.txt": "# LayerPrunedFC_NN Hyperparameter Tuning Jobs",
        "relaxed_global_prune_nn_warmup_jobs.txt": "# RelaxedGlobalPrunedFC_NN Hyperparameter Tuning Jobs (relaxed pruning)",
        "uniform_random_do_di_snn_jobs.txt": "# Uniform Random DO+DI (Direct Output + Direct Input) Sparse NN Hyperparameter Tuning Jobs",
        "ERK_SNN_jobs.txt": "# ERK_SNN Hyperparameter Tuning Jobs",
    }
    for rel_out, rel_in, pair in JOBSPECS:
        src = SCHEDULER / rel_in
        dst = APE / rel_out
        if not src.is_file():
            raise FileNotFoundError(src)
        name = Path(rel_out).name
        title = titles.get(name, f"# {name}")
        process_file(src, dst, pair, title)

    json_roots = {
        "fnest_nn_advanced_jobs.json": "fNeST-NN Hyperparameter Tuning Jobs",
        "dense_fnest_advanced_jobs.json": "Dense_fNeST Hyperparameter Tuning Jobs",
        "fc_nn_advanced_jobs.json": "FC_NN Hyperparameter Tuning Jobs",
        "r_sparse_nn_advanced_jobs.json": "Sparse_NN Hyperparameter Tuning Jobs",
        "nest_vnn_advanced_jobs.json": "NeST VNN Hyperparameter Tuning Jobs",
        "global_prune_nn_warmup_advanced_jobs.json": "GlobalPrunedFC_NN Hyperparameter Tuning Jobs (with warmup)",
        "layer_prune_nn_advanced_jobs.json": "LayerPrunedFC_NN Hyperparameter Tuning Jobs",
        "relaxed_global_prune_nn_warmup_advanced_jobs.json": "RelaxedGlobalPrunedFC_NN Hyperparameter Tuning Jobs (relaxed pruning)",
        "uniform_random_do_di_snn_advanced_jobs.json": "Uniform Random DO+DI (Direct Output + Direct Input) Sparse NN Hyperparameter Tuning Jobs",
        "ERK_SNN_advanced_jobs.json": "ERK_SNN Hyperparameter Tuning Jobs",
    }
    for rel_out, rel_in, pair in JSONSPECS:
        src = SCHEDULER / rel_in
        dst = APE / rel_out
        if not src.is_file():
            raise FileNotFoundError(src)
        process_json(src, dst, pair, json_roots.get(Path(rel_out).name))


if __name__ == "__main__":
    main()
