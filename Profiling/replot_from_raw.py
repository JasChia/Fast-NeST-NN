"""Rebuild summary_statistics.csv and the Figure 2 bar chart from saved raw timings.

profile_networks.py only writes the figure at the end of a fresh profiling run. This reads
the two *_raw_times.json it saved (key `backward_times` holds the training-iteration times:
forward + loss + backward, no optimizer step) and calls the same compute_statistics /
create_relative_speedup_plot, so the published numbers can be regenerated without a GPU.

    python replot_from_raw.py --npa-dir profiling_results/npa_4 --output-dir replot
"""
import argparse, json
from pathlib import Path
import profile_networks as pn

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--npa-dir", type=Path, default=Path("profiling_results/npa_4"))
ap.add_argument("--output-dir", type=Path, required=True)
a = ap.parse_args()
a.output_dir.mkdir(parents=True, exist_ok=True)
enest = json.loads((a.npa_dir / "eNest_raw_times.json").read_text())
dc = json.loads((a.npa_dir / "DrugCellNN_raw_times.json").read_text())
summary = pn.compute_statistics(enest, dc, str(a.output_dir))
pn.create_relative_speedup_plot(summary, str(a.output_dir))
print(f"wrote summary and {pn.PAPER_SPEEDUP_BAR_FIGURE} to {a.output_dir}")
