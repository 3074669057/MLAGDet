"""Regenerate scalability_summary.md from existing CSV."""
import sys
from pathlib import Path

import pandas as pd

_EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXPERIMENT_DIR))

from scalability_efficiency_profiler import (  # noqa: E402
    OUTPUT_ROOT,
    ProfileRow,
    collect_environment,
    write_summary_md,
)

csv_path = OUTPUT_ROOT / "scalability_profile_results.csv"
df = pd.read_csv(csv_path)
rows = []
for _, r in df.iterrows():
    rt = r["runtime_seconds"]
    rt = None if rt == "not_available" else float(rt)
    pm = r["peak_memory_mb"]
    pm = None if pm == "not_available" else float(pm)
    oc = r["output_count"]
    oc = None if pd.isna(oc) or oc == "" else int(oc)
    rows.append(
        ProfileRow(
            scale_fraction=float(r["scale_fraction"]),
            tx_count=int(r["tx_count"]),
            node_count=int(r["node_count"]),
            edge_count=int(r["edge_count"]),
            module=str(r["module"]),
            algorithm=str(r["algorithm"]),
            runtime_seconds=rt,
            peak_memory_mb=pm,
            output_count=oc,
            notes=str(r["notes"]) if not pd.isna(r["notes"]) else "",
        )
    )

scale_stats = []
for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
    sub = df[df["scale_fraction"] == frac]
    if sub.empty:
        continue
    r0 = sub.iloc[0]
    pt = sub[(sub["module"] == "point_level_detection") & (sub["algorithm"] == "point_level_total")]
    comb = sub[
        (sub["module"] == "line_level_partitioning") & (sub["algorithm"] == "combined")
    ]
    scale_stats.append(
        {
            "fraction": frac,
            "tx_count": int(r0["tx_count"]),
            "node_count": int(r0["node_count"]),
            "edge_count": int(r0["edge_count"]),
            "suspicious": int(pt["output_count"].iloc[0]) if len(pt) else "N/A",
            "combined_communities": int(comb["output_count"].iloc[0]) if len(comb) else "N/A",
        }
    )

write_summary_md(
    collect_environment(),
    scale_stats,
    rows,
    OUTPUT_ROOT / "scalability_summary.md",
    "Skipped run_analysis.py / Scrapy online APPR (RPC + Etherscan + network I/O). "
    "Offline personalized PageRank only (phase1_offline_appr_seed_degradation).",
    "Stage-3 skipped in batch profiler; common_txs ~168k rows — profile separately.",
    "- stage1_total: full subprocess pipeline including save_results and ROC plots.\n"
    "- Module breakdown: instrumented path skips save_results/ROC; sum < stage1_total.\n"
    "- line_level per-algorithm peak_memory: not_available (no per-step monitor).\n"
    "- offline APPR: some large seed communities hit max_subgraph_nodes=50000.",
)
print("Wrote", OUTPUT_ROOT / "scalability_summary.md")
