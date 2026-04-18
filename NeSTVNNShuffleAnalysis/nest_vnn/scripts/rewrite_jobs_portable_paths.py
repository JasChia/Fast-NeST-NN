#!/usr/bin/env python3
"""Rewrite nest_vnn_jobs/jobs.txt for portable paths (Data/, nest_vnn_logs/)."""
import re
from pathlib import Path

OLD_DATA_ROOT = "/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/jchia_backup/ANL_Drug_CData"
NEW_DATA_ROOT = "../Data"  # relative to nest_vnn/ working directory

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "nest_vnn_jobs" / "jobs.txt"


def main():
    text = JOBS.read_text(encoding="utf-8")
    text = text.replace(OLD_DATA_ROOT, NEW_DATA_ROOT)
    # model_D5_unshuffled_0 -> nest_vnn_logs/D5_unshuffled_exp_0 (combine_metrics / paper layout)
    text = re.sub(
        r"-modeldir model_D(\d+)_(unshuffled|shuffled)_(\d+)",
        r"-modeldir nest_vnn_logs/D\1_\2_exp_\3",
        text,
    )
    JOBS.write_text(text, encoding="utf-8")
    print(f"Updated {JOBS}")


if __name__ == "__main__":
    main()
