#!/usr/bin/env python3
"""
Phase-1 seed degradation experiment: simulate point-level false negatives by
removing known malicious seeds from line-level communities, then measure whether
surface-level APPR expansion can recover them via remaining seeds.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMUNITY_CONTAINER_KEYS = frozenset(
    {"nodes", "accounts", "members", "suspicious_accounts", "community", "addresses"}
)
ADDRESS_ITEM_KEYS = frozenset(
    {"address", "node", "id", "account", "from", "to", "source", "out"}
)
ACCOUNT_COLUMN_PRIORITY = ("node", "address", "account", "id")
SCORE_COLUMN_PRIORITY = (
    "importance",
    "score",
    "p",
    "pagerank",
    "appr_score",
    "weight",
)
LABEL_ADDRESS_COLUMNS = ("address", "node", "id", "account")
ETH_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

DEFAULT_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_TOPK = [20, 50, 100, 200]
TRIAL_CSV = "phase1_seed_degradation_trial_results.csv"
SUMMARY_CSV = "phase1_seed_degradation_summary.csv"
PLOT_PNG = "phase1_seed_degradation_plot.png"
LOG_TXT = "phase1_seed_degradation_log.txt"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(relative: str) -> Path:
    """Resolve a project-relative path against PROJECT_ROOT."""
    path = Path(relative)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


# ---------------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------------


def normalize_address(value: Any) -> Optional[str]:
    """Normalize an Ethereum address to lowercase 0x-prefixed 40-hex form."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text.startswith("0x"):
        return None
    # Keep only valid 42-char addresses; trim longer polluted strings.
    if len(text) >= 42:
        candidate = text[:42]
        if ETH_ADDRESS_RE.match(candidate):
            return candidate
    return None


def normalize_address_set(values: Iterable[Any]) -> Set[str]:
    """Collect normalized addresses from an iterable."""
    result: Set[str] = set()
    for v in values:
        addr = normalize_address(v)
        if addr:
            result.add(addr)
    return result


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------


def _detect_address_column(df: pd.DataFrame) -> Optional[str]:
    """Pick the best address column from a label dataframe."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in LABEL_ADDRESS_COLUMNS:
        if name in lower_map:
            return lower_map[name]
    if len(df.columns) >= 1:
        return df.columns[0]
    return None


def load_label_addresses(path: Path, label_name: str) -> Set[str]:
    """Load and normalize addresses from a CE/BE CSV file."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label_name} label file: {path}\n"
            f"Expected malicious/benign labels at labels/CE.csv and labels/BE.csv, "
            f"or pass --labels_ce / --labels_be."
        )
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Cannot read {label_name} file {path}: {exc}") from exc

    if df.empty:
        logging.warning("%s file is empty: %s", label_name, path)
        return set()

    col = _detect_address_column(df)
    if col is None:
        raise ValueError(f"No usable address column in {path}")

    addresses = set()
    for raw in df[col].dropna().astype(str):
        addr = normalize_address(raw)
        if addr:
            addresses.add(addr)
    return addresses


# ---------------------------------------------------------------------------
# Community JSON parsing
# ---------------------------------------------------------------------------


def _extract_addresses_from_item(item: Any, out: Set[str]) -> None:
    """Extract address from a single dict/list/scalar node."""
    if isinstance(item, dict):
        for key, val in item.items():
            key_l = str(key).lower()
            if key_l in COMMUNITY_CONTAINER_KEYS:
                _extract_addresses_from_container(val, out)
            elif key_l in ADDRESS_ITEM_KEYS:
                addr = normalize_address(val)
                if addr:
                    out.add(addr)
            else:
                _extract_addresses_from_item(val, out)
    elif isinstance(item, (list, tuple, set)):
        for sub in item:
            _extract_addresses_from_item(sub, out)
    else:
        addr = normalize_address(item)
        if addr:
            out.add(addr)


def _extract_addresses_from_container(data: Any, out: Set[str]) -> None:
    """Extract addresses from a community container field."""
    if isinstance(data, dict):
        for key in COMMUNITY_CONTAINER_KEYS:
            if key in data:
                _extract_addresses_from_container(data[key], out)
        _extract_addresses_from_item(data, out)
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            _extract_addresses_from_item(item, out)
    else:
        _extract_addresses_from_item(data, out)


def parse_community_accounts(json_path: Path) -> Set[str]:
    """
    Robustly parse community JSON and return the account set.
    Supports dict/list hybrids and multiple field naming conventions.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {json_path}: {exc}") from exc

    accounts: Set[str] = set()

    if isinstance(data, dict):
        for key in COMMUNITY_CONTAINER_KEYS:
            if key in data:
                _extract_addresses_from_container(data[key], accounts)
        _extract_addresses_from_item(data, accounts)
    elif isinstance(data, list):
        for item in data:
            _extract_addresses_from_item(item, accounts)
    else:
        _extract_addresses_from_item(data, accounts)

    return accounts


def discover_community_files(communities_dir: Path) -> List[Path]:
    """Find all community JSON files under the given directory."""
    if not communities_dir.is_dir():
        raise FileNotFoundError(
            f"Communities directory not found: {communities_dir}\n"
            f"Run the main pipeline first to produce results1/combined/*.json, "
            f"or pass --communities_dir."
        )
    files = sorted(communities_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON community files in {communities_dir}\n"
            f"Specify an alternate directory with --communities_dir."
        )
    return files


# ---------------------------------------------------------------------------
# APPR importance loading
# ---------------------------------------------------------------------------


def _detect_column(df: pd.DataFrame, priority: Tuple[str, ...]) -> Optional[str]:
    """Return the first matching column name (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in priority:
        if name in lower_map:
            return lower_map[name]
    return None


def _extract_seed_from_path(csv_path: Path) -> Optional[str]:
    """
    Infer seed address from importance CSV path or parent directory.
    Example: Spider/0xabc/importance/0xabc.csv -> 0xabc
    """
    stem = normalize_address(csv_path.stem)
    if stem:
        return stem
    parent = normalize_address(csv_path.parent.parent.name)
    if parent:
        return parent
    # Walk upward for any 0x directory component.
    for part in reversed(csv_path.parts):
        addr = normalize_address(part)
        if addr:
            return addr
    return None


def load_importance_file(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Parse one APPR importance CSV into columns: node, score, rank.
    Returns None if the file cannot be parsed.
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as exc:
        logging.warning("Skip unreadable importance file %s: %s", csv_path, exc)
        return None

    if df.empty:
        logging.warning("Skip empty importance file: %s", csv_path)
        return None

    node_col = _detect_column(df, ACCOUNT_COLUMN_PRIORITY)
    if node_col is None:
        logging.warning("Skip %s: no account column found", csv_path)
        return None

    score_col = _detect_column(df, SCORE_COLUMN_PRIORITY)
    rows = []
    for i, row in df.iterrows():
        node = normalize_address(row[node_col])
        if not node:
            continue
        if score_col is not None:
            try:
                score = float(row[score_col])
            except (TypeError, ValueError):
                score = 1.0 / (float(i) + 1.0)
        else:
            score = 1.0 / (float(i) + 1.0)
        rows.append({"node": node, "score": score})

    if not rows:
        logging.warning("Skip %s: no valid node rows", csv_path)
        return None

    out = pd.DataFrame(rows)
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def discover_importance_files(spider_dir: Path) -> List[Path]:
    """Recursively find all importance/*.csv under Spider."""
    if not spider_dir.is_dir():
        raise FileNotFoundError(
            f"Spider directory not found: {spider_dir}\n"
            f"APPR importance CSVs are produced by run_analysis.py + Scrapy. "
            f"Pass --spider_dir if stored elsewhere."
        )
    files = sorted(spider_dir.rglob("importance/*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No importance CSV files under {spider_dir}/**/importance/\n"
            f"Run run_analysis.py on community JSONs first; this script only reads "
            f"existing APPR outputs."
        )
    return files


def build_seed_to_ranked_candidates(
    importance_paths: List[Path],
) -> Dict[str, pd.DataFrame]:
    """Map each seed address to its ranked APPR candidate dataframe."""
    seed_map: Dict[str, pd.DataFrame] = {}
    for path in importance_paths:
        seed = _extract_seed_from_path(path)
        if seed is None:
            logging.warning("Cannot infer seed from path, skip: %s", path)
            continue
        ranked = load_importance_file(path)
        if ranked is None:
            continue
        # If duplicate seed files exist, keep the one with more candidates.
        if seed not in seed_map or len(ranked) > len(seed_map[seed]):
            seed_map[seed] = ranked
    return seed_map


# ---------------------------------------------------------------------------
# Candidate merge & metrics
# ---------------------------------------------------------------------------


def merge_appr_candidates(
    remaining_seeds: Set[str],
    seed_to_candidates: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, int]:
    """
    Merge APPR lists from remaining seeds with per-seed score normalization.
    Returns (merged_df, available_seed_file_count).
    """
    available_seeds = [s for s in remaining_seeds if s in seed_to_candidates]
    if not available_seeds:
        empty = pd.DataFrame(
            columns=["node", "aggregate_score", "hit_count", "consistency", "best_rank"]
        )
        return empty, 0

    agg: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"aggregate_score": 0.0, "hit_count": 0, "best_rank": float("inf")}
    )

    for seed in available_seeds:
        df = seed_to_candidates[seed]
        max_score = float(df["score"].max())
        if max_score <= 0:
            max_score = 1.0
        for _, row in df.iterrows():
            node = row["node"]
            norm_score = float(row["score"]) / max_score
            rank = float(row["rank"])
            entry = agg[node]
            entry["aggregate_score"] += norm_score
            entry["hit_count"] += 1
            entry["best_rank"] = min(entry["best_rank"], rank)

    n_files = len(available_seeds)
    records = []
    for node, vals in agg.items():
        records.append(
            {
                "node": node,
                "aggregate_score": vals["aggregate_score"],
                "hit_count": vals["hit_count"],
                "consistency": vals["hit_count"] / n_files,
                "best_rank": vals["best_rank"],
            }
        )

    merged = pd.DataFrame(records)
    merged = merged.sort_values(
        by=["aggregate_score", "consistency", "best_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    merged["merged_rank"] = merged.index + 1
    return merged, n_files


def removal_count(malicious_count: int, ratio: float) -> int:
    """How many malicious seeds to remove for a given ratio."""
    if ratio <= 0.0 or malicious_count == 0:
        return 0
    return max(1, int(round(ratio * malicious_count)))


def compute_trial_metrics(
    merged: pd.DataFrame,
    removed_seeds: Set[str],
    ce_labels: Set[str],
    topk_list: List[int],
    ratio: float,
    removed_n: int,
    remaining_seed_count: int,
    available_seed_file_count: int,
) -> List[Dict[str, Any]]:
    """Compute per-K metrics for one trial."""
    rank_lookup = (
        dict(zip(merged["node"], merged["merged_rank"])) if not merged.empty else {}
    )
    consistency_lookup = (
        dict(zip(merged["node"], merged["consistency"])) if not merged.empty else {}
    )
    ordered_nodes = merged["node"].tolist() if not merged.empty else []
    candidate_count = len(ordered_nodes)

    results = []
    for k in topk_list:
        top_nodes = set(ordered_nodes[:k])
        recovered = removed_seeds & top_nodes
        recovered_n = len(recovered)

        if ratio == 0.0 or removed_n == 0:
            recovery_rate = float("nan")
        else:
            recovery_rate = recovered_n / removed_n

        ce_hits = len(top_nodes & ce_labels)

        if recovered:
            ranks = [rank_lookup[n] for n in recovered]
            consistencies = [consistency_lookup[n] for n in recovered]
            mean_rank = float(np.mean(ranks))
            mean_consistency = float(np.mean(consistencies))
        else:
            mean_rank = float("nan")
            mean_consistency = float("nan")

        results.append(
            {
                "removed_recovery_at_k": recovery_rate,
                "removed_count": removed_n,
                "recovered_removed_count_at_k": recovered_n,
                "ce_hits_at_k": ce_hits,
                "candidate_count": candidate_count,
                "mean_rank_of_recovered_removed": mean_rank,
                "mean_consistency_of_recovered_removed": mean_consistency,
                "remaining_seed_count": remaining_seed_count,
                "available_seed_file_count": available_seed_file_count,
                "top_k": k,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Community eligibility & experiment
# ---------------------------------------------------------------------------


def load_communities(
    community_files: List[Path],
    ce_labels: Set[str],
    seed_to_candidates: Dict[str, pd.DataFrame],
    min_malicious_seeds: int,
    min_appr_seeds: int = 2,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Parse communities and filter those meeting experiment criteria.
    Returns (eligible_communities, skip_log_lines).
    """
    eligible: List[Dict[str, Any]] = []
    skip_logs: List[str] = []

    for path in community_files:
        community_id = path.stem
        try:
            accounts = parse_community_accounts(path)
        except Exception as exc:
            skip_logs.append(f"{community_id}: parse error - {exc}")
            continue

        if not accounts:
            skip_logs.append(f"{community_id}: no accounts extracted")
            continue

        malicious_seeds = accounts & ce_labels
        if len(malicious_seeds) < min_malicious_seeds:
            skip_logs.append(
                f"{community_id}: malicious_seeds={len(malicious_seeds)} "
                f"< min {min_malicious_seeds}"
            )
            continue

        appr_available = [s for s in malicious_seeds if s in seed_to_candidates]
        if len(appr_available) < min_appr_seeds:
            skip_logs.append(
                f"{community_id}: appr_files_for_malicious={len(appr_available)} "
                f"< min {min_appr_seeds}"
            )
            continue

        eligible.append(
            {
                "community_id": community_id,
                "path": path,
                "accounts": accounts,
                "malicious_seeds": malicious_seeds,
                "appr_available_malicious": set(appr_available),
            }
        )

    return eligible, skip_logs


def run_experiments(
    eligible: List[Dict[str, Any]],
    seed_to_candidates: Dict[str, pd.DataFrame],
    ce_labels: Set[str],
    ratios: List[float],
    topk_list: List[int],
    repeats: int,
    random_seed: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run all degradation trials; return trial dataframe and skip logs."""
    rng = random.Random(random_seed)
    trial_rows: List[Dict[str, Any]] = []
    skip_logs: List[str] = []

    for comm in eligible:
        community_id = comm["community_id"]
        malicious_seeds = sorted(comm["malicious_seeds"])
        malicious_set = set(malicious_seeds)

        for ratio in ratios:
            n_remove = removal_count(len(malicious_seeds), ratio)

            for repeat_idx in range(repeats):
                if n_remove == 0:
                    removed = set()
                else:
                    removed = set(rng.sample(malicious_seeds, n_remove))

                remaining = malicious_set - removed
                merged, avail_files = merge_appr_candidates(
                    remaining, seed_to_candidates
                )

                if avail_files == 0:
                    skip_logs.append(
                        f"{community_id} ratio={ratio} repeat={repeat_idx}: "
                        "no APPR files for remaining seeds"
                    )
                    continue

                metrics_list = compute_trial_metrics(
                    merged=merged,
                    removed_seeds=removed,
                    ce_labels=ce_labels,
                    topk_list=topk_list,
                    ratio=ratio,
                    removed_n=len(removed),
                    remaining_seed_count=len(remaining),
                    available_seed_file_count=avail_files,
                )

                for m in metrics_list:
                    row = {
                        "community_id": community_id,
                        "removal_ratio": ratio,
                        "repeat": repeat_idx,
                        **m,
                    }
                    trial_rows.append(row)

    trials_df = pd.DataFrame(trial_rows)
    return trials_df, skip_logs


def build_summary(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial results by removal_ratio and top_k."""
    if trials_df.empty:
        return pd.DataFrame()

    agg_spec = {
        "removed_recovery_at_k": ["mean", "std"],
        "ce_hits_at_k": ["mean", "std"],
        "candidate_count": ["mean"],
        "remaining_seed_count": ["mean"],
        "available_seed_file_count": ["mean"],
        "mean_rank_of_recovered_removed": ["mean"],
        "mean_consistency_of_recovered_removed": ["mean"],
    }

    grouped = trials_df.groupby(["removal_ratio", "top_k"], dropna=False).agg(agg_spec)
    grouped.columns = [
        "removed_recovery_at_k_mean",
        "removed_recovery_at_k_std",
        "ce_hits_at_k_mean",
        "ce_hits_at_k_std",
        "candidate_count_mean",
        "remaining_seed_count_mean",
        "available_seed_file_count_mean",
        "mean_rank_of_recovered_removed_mean",
        "mean_consistency_of_recovered_removed_mean",
    ]
    return grouped.reset_index()


def save_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot mean recovery rate vs removal ratio for K=50,100,200."""
    if summary_df.empty:
        logging.warning("Summary empty; skip plot generation.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_ks = [50, 100, 200]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for k, color in zip(plot_ks, colors):
        sub = summary_df[summary_df["top_k"] == k].sort_values("removal_ratio")
        if sub.empty:
            continue
        ax.plot(
            sub["removal_ratio"],
            sub["removed_recovery_at_k_mean"],
            marker="o",
            label=f"K={k}",
            color=color,
        )

    ax.set_title(
        "Robustness of APPR Expansion under Simulated Point-Level False Negatives"
    )
    ax.set_xlabel("Simulated false-negative ratio")
    ax.set_ylabel("Recovery rate of removed malicious seeds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_log(
    log_path: Path,
    community_files: List[Path],
    eligible: List[Dict[str, Any]],
    skip_logs: List[str],
    trial_skip_logs: List[str],
    importance_count: int,
    ce_count: int,
    be_count: int,
    dry_run: bool,
) -> None:
    """Write experiment log file."""
    lines = [
        "Phase-1 Seed Degradation Experiment Log",
        "=" * 50,
        f"dry_run: {dry_run}",
        f"community_json_files: {len(community_files)}",
        f"eligible_communities: {len(eligible)}",
        f"appr_importance_files: {importance_count}",
        f"ce_label_count: {ce_count}",
        f"be_label_count: {be_count}",
        "",
        "Eligible communities:",
    ]
    for comm in eligible:
        lines.append(
            f"  - {comm['community_id']}: "
            f"malicious_seeds={len(comm['malicious_seeds'])}, "
            f"appr_malicious={len(comm['appr_available_malicious'])}"
        )

    lines.extend(["", "Skipped communities (filter stage):"])
    if skip_logs:
        lines.extend(f"  - {s}" for s in skip_logs)
    else:
        lines.append("  (none)")

    lines.extend(["", "Skipped trials (runtime):"])
    if trial_skip_logs:
        lines.extend(f"  - {s}" for s in trial_skip_logs)
    else:
        lines.append("  (none)")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-1 experiment: simulate point-level false negatives by removing "
            "malicious seeds and measure APPR recovery via remaining seeds."
        )
    )
    parser.add_argument(
        "--communities_dir",
        default="results1/combined",
        help="Directory with community JSON files (default: results1/combined)",
    )
    parser.add_argument(
        "--labels_ce",
        default="labels/CE.csv",
        help="Malicious (CE) label CSV (default: labels/CE.csv)",
    )
    parser.add_argument(
        "--labels_be",
        default="labels/BE.csv",
        help="Benign (BE) label CSV (default: labels/BE.csv)",
    )
    parser.add_argument(
        "--spider_dir",
        default="Spider",
        help="Spider root directory for APPR importance CSVs (default: Spider)",
    )
    parser.add_argument(
        "--output_dir",
        default="phase1/seed_degradation",
        help="Output directory (default: phase1/seed_degradation)",
    )
    parser.add_argument(
        "--ratios",
        default="0,0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated removal ratios (default: 0,0.1,...,0.5)",
    )
    parser.add_argument(
        "--topk",
        default="20,50,100,200",
        help="Comma-separated top-K values (default: 20,50,100,200)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Random repeats per community/ratio (default: 20)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="RNG seed for reproducible removals (default: 42)",
    )
    parser.add_argument(
        "--min_malicious_seeds",
        type=int,
        default=3,
        help="Minimum malicious seeds per community (default: 3)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only validate inputs and list eligible communities",
    )
    return parser.parse_args()


def _parse_float_list(text: str, name: str) -> List[float]:
    try:
        return [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {text}") from exc


def _parse_int_list(text: str, name: str) -> List[int]:
    try:
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {text}") from exc


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    communities_dir = resolve_project_path(args.communities_dir)
    labels_ce = resolve_project_path(args.labels_ce)
    labels_be = resolve_project_path(args.labels_be)
    spider_dir = resolve_project_path(args.spider_dir)
    output_dir = resolve_project_path(args.output_dir)
    ratios = _parse_float_list(args.ratios, "ratios")
    topk_list = _parse_int_list(args.topk, "topk")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load inputs ---
    ce_labels = load_label_addresses(labels_ce, "CE")
    be_labels = load_label_addresses(labels_be, "BE")
    community_files = discover_community_files(communities_dir)
    importance_paths = discover_importance_files(spider_dir)
    seed_to_candidates = build_seed_to_ranked_candidates(importance_paths)

    eligible, filter_skip_logs = load_communities(
        community_files=community_files,
        ce_labels=ce_labels,
        seed_to_candidates=seed_to_candidates,
        min_malicious_seeds=args.min_malicious_seeds,
    )

    logging.info("Community JSON files: %d", len(community_files))
    logging.info("Eligible communities: %d", len(eligible))
    logging.info("APPR importance files: %d", len(importance_paths))
    logging.info("CE labels: %d | BE labels: %d", len(ce_labels), len(be_labels))

    if args.dry_run:
        write_log(
            log_path=output_dir / LOG_TXT,
            community_files=community_files,
            eligible=eligible,
            skip_logs=filter_skip_logs,
            trial_skip_logs=[],
            importance_count=len(importance_paths),
            ce_count=len(ce_labels),
            be_count=len(be_labels),
            dry_run=True,
        )
        print(f"\n[DRY RUN] Output directory: {output_dir.resolve()}")
        print(f"[DRY RUN] Valid communities: {len(eligible)} / {len(community_files)}")
        print(f"[DRY RUN] Log: {(output_dir / LOG_TXT).resolve()}")
        if not eligible:
            print(
                "\nWARNING: No eligible communities. Need >=3 malicious seeds per "
                "community and >=2 with APPR importance files."
            )
        return

    if not eligible:
        write_log(
            log_path=output_dir / LOG_TXT,
            community_files=community_files,
            eligible=eligible,
            skip_logs=filter_skip_logs,
            trial_skip_logs=[],
            importance_count=len(importance_paths),
            ce_count=len(ce_labels),
            be_count=len(be_labels),
            dry_run=False,
        )
        raise RuntimeError(
            "No eligible communities for experiment. See "
            f"{output_dir / LOG_TXT} for skip reasons."
        )

    trials_df, trial_skip_logs = run_experiments(
        eligible=eligible,
        seed_to_candidates=seed_to_candidates,
        ce_labels=ce_labels,
        ratios=ratios,
        topk_list=topk_list,
        repeats=args.repeats,
        random_seed=args.random_seed,
    )

    summary_df = build_summary(trials_df)

    trial_path = output_dir / TRIAL_CSV
    summary_path = output_dir / SUMMARY_CSV
    plot_path = output_dir / PLOT_PNG
    log_path = output_dir / LOG_TXT

    trials_df.to_csv(trial_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    save_plot(summary_df, plot_path)
    write_log(
        log_path=log_path,
        community_files=community_files,
        eligible=eligible,
        skip_logs=filter_skip_logs,
        trial_skip_logs=trial_skip_logs,
        importance_count=len(importance_paths),
        ce_count=len(ce_labels),
        be_count=len(be_labels),
        dry_run=False,
    )

    print(f"\nOutput directory: {output_dir.resolve()}")
    print(f"Valid communities: {len(eligible)}")
    print(f"Summary CSV: {summary_path.resolve()}")
    print(f"Plot: {plot_path.resolve()}")
    print(f"Trial results: {trial_path.resolve()}")
    print(f"Log: {log_path.resolve()}")


if __name__ == "__main__":
    main()
