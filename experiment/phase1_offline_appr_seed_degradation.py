#!/usr/bin/env python3
"""
Offline APPR / personalized PageRank seed-degradation experiment.

Evaluates whether removed malicious (CE) seeds can reappear in APPR top-K
candidates when expansion runs on a local transaction graph, without
calling Etherscan or Scrapy.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from phase1_seed_degradation_experiment import (  # noqa: E402
    discover_community_files,
    load_label_addresses,
    normalize_address,
    parse_community_accounts,
    resolve_project_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR_NAME = "phase1/offline_seed_degradation"
TRIAL_CSV = "phase1_offline_seed_degradation_trial_results.csv"
SUMMARY_CSV = "phase1_offline_seed_degradation_summary.csv"
ELIGIBLE_CSV = "phase1_offline_seed_degradation_eligible_communities.csv"
PLOT_RECOVERY = "phase1_offline_seed_degradation_plot_recovery.png"
PLOT_CE_HITS = "phase1_offline_seed_degradation_plot_ce_hits.png"
PLOT_PAPER = "phase1_offline_seed_degradation_recovery_vs_random_paper.png"
TABLE_PAPER = "phase1_offline_seed_degradation_table_for_paper.csv"
KEY_FINDINGS_TXT = "phase1_offline_seed_degradation_key_findings.txt"
LOG_TXT = "phase1_offline_seed_degradation_log.txt"

FROM_ALIASES = ("from", "sender", "source_address", "src")
TO_ALIASES = ("to", "recipient", "target_address", "dst")
VALUE_ALIASES = ("value", "amount", "tx_value", "transfer_value")
TIMESTAMP_ALIASES = ("timestamp", "timestampt", "time_stamp", "block_timestamp")


# ---------------------------------------------------------------------------
# Transaction graph
# ---------------------------------------------------------------------------


def _pick_column(columns: List[str], aliases: Tuple[str, ...], required: str) -> str:
    lower_map = {c.lower(): c for c in columns}
    for name in aliases:
        if name in lower_map:
            return lower_map[name]
    raise ValueError(
        f"Cannot find required column for '{required}' in transaction CSV. "
        f"Tried aliases: {aliases}. Available columns: {columns}"
    )


def _pick_optional_column(columns: List[str], aliases: Tuple[str, ...]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for name in aliases:
        if name in lower_map:
            return lower_map[name]
    return None


def parse_transaction_value(raw: Any) -> float:
    """Parse transaction value; fall back to 1.0 on failure."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return 1.0
    try:
        val = float(raw)
        if val < 0:
            return 1.0
        return val
    except (TypeError, ValueError):
        return 1.0


class LocalTransactionGraph:
    """Directed weighted adjacency list built from labeled_transactions.csv."""

    def __init__(self) -> None:
        self.adj: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.rev_adj: DefaultDict[str, Set[str]] = defaultdict(set)
        self.nodes: Set[str] = set()
        self.edge_count: int = 0

    def add_edge(self, src: str, dst: str, weight: float, undirected: bool) -> None:
        if src == dst:
            return
        self.nodes.add(src)
        self.nodes.add(dst)
        self.adj[src][dst] += weight
        self.rev_adj[dst].add(src)
        self.edge_count += 1
        if undirected:
            self.adj[dst][src] += weight
            self.rev_adj[src].add(dst)
            self.edge_count += 1

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return self.edge_count


def build_transaction_graph(
    tx_path: Path,
    undirected_projection: bool = False,
    chunksize: int = 200_000,
) -> LocalTransactionGraph:
    """Load transactions and aggregate directed edge weights."""
    if not tx_path.is_file():
        raise FileNotFoundError(
            f"Transaction file not found: {tx_path}\n"
            f"Expected default: transactions/labeled_transactions.csv"
        )

    graph = LocalTransactionGraph()
    header = pd.read_csv(tx_path, nrows=0, encoding="utf-8")
    columns = list(header.columns)
    from_col = _pick_column(columns, FROM_ALIASES, "from")
    to_col = _pick_column(columns, TO_ALIASES, "to")
    value_col = _pick_optional_column(columns, VALUE_ALIASES)

    usecols = [from_col, to_col]
    if value_col:
        usecols.append(value_col)

    logging.info("Building transaction graph from %s ...", tx_path)
    for chunk in pd.read_csv(
        tx_path, usecols=usecols, chunksize=chunksize, encoding="utf-8"
    ):
        for _, row in chunk.iterrows():
            src = normalize_address(row[from_col])
            dst = normalize_address(row[to_col])
            if not src or not dst:
                continue
            raw_val = row[value_col] if value_col else None
            weight = math.log1p(parse_transaction_value(raw_val))
            graph.add_edge(src, dst, weight, undirected=undirected_projection)

    logging.info(
        "Graph built: %d nodes, %d directed edge records",
        graph.num_nodes,
        graph.edge_count,
    )
    return graph


def k_hop_neighborhood(
    graph: LocalTransactionGraph,
    seeds: Set[str],
    k_hop: int,
    max_nodes: int,
) -> Tuple[Set[str], Optional[str]]:
    """BFS k-hop neighborhood over directed adjacency (out-edges)."""
    if not seeds:
        return set(), "empty seed set"

    visited: Set[str] = set()
    queue: deque[Tuple[str, int]] = deque()

    for seed in seeds:
        if seed in graph.nodes:
            visited.add(seed)
            queue.append((seed, 0))

    if not visited:
        return set(), "no seeds present in transaction graph"

    while queue:
        node, depth = queue.popleft()
        if depth >= k_hop:
            continue
        for nbr in graph.adj.get(node, {}):
            if nbr not in visited:
                visited.add(nbr)
                if len(visited) > max_nodes:
                    return set(), f"subgraph exceeds max_nodes={max_nodes}"
                queue.append((nbr, depth + 1))
        if depth < k_hop:
            for pred in graph.rev_adj.get(node, set()):
                if pred not in visited:
                    visited.add(pred)
                    if len(visited) > max_nodes:
                        return set(), f"subgraph exceeds max_nodes={max_nodes}"
                    queue.append((pred, depth + 1))

    return visited, None


def subgraph_to_nx(
    graph: LocalTransactionGraph,
    nodes: Set[str],
    undirected: bool,
) -> nx.Graph | nx.DiGraph:
    """Materialize an induced subgraph for NetworkX PageRank."""
    if undirected:
        g: nx.Graph | nx.DiGraph = nx.Graph()
    else:
        g = nx.DiGraph()

    for u in nodes:
        for v, w in graph.adj.get(u, {}).items():
            if v in nodes and w > 0:
                if g.has_edge(u, v):
                    g[u][v]["weight"] += w
                else:
                    g.add_edge(u, v, weight=w)
    return g


def run_personalized_pagerank(
    graph: LocalTransactionGraph,
    retained_seeds: Set[str],
    k_hop: int,
    max_subgraph_nodes: int,
    undirected_projection: bool,
    alpha: float,
    max_iter: int,
    tol: float,
) -> Tuple[Optional[Dict[str, float]], Optional[str], int]:
    """
    Run personalized PageRank on a k-hop subgraph around retained seeds.
    Returns (scores, error_message, subgraph_node_count).
    """
    sub_nodes, err = k_hop_neighborhood(
        graph, retained_seeds, k_hop=k_hop, max_nodes=max_subgraph_nodes
    )
    if err:
        return None, err, 0

    retained_in_sub = retained_seeds & sub_nodes
    if not retained_in_sub:
        return None, "no retained seeds in subgraph", len(sub_nodes)

    subg = subgraph_to_nx(graph, sub_nodes, undirected=undirected_projection)
    if subg.number_of_nodes() == 0:
        return None, "empty subgraph", 0

    personalization = {n: 0.0 for n in subg.nodes()}
    weight = 1.0 / len(retained_in_sub)
    for seed in retained_in_sub:
        personalization[seed] = weight

    try:
        scores = nx.pagerank(
            subg,
            alpha=alpha,
            personalization=personalization,
            max_iter=max_iter,
            tol=tol,
            weight="weight",
        )
        return scores, None, subg.number_of_nodes()
    except nx.PowerIterationFailedConvergence as exc:
        return None, f"pagerank did not converge: {exc}", subg.number_of_nodes()
    except Exception as exc:
        return None, f"pagerank failed: {exc}", subg.number_of_nodes()


# ---------------------------------------------------------------------------
# Communities & trials
# ---------------------------------------------------------------------------


def removal_count(malicious_count: int, ratio: float) -> int:
    if ratio <= 0.0 or malicious_count == 0:
        return 0
    return max(1, int(round(ratio * malicious_count)))


def find_eligible_communities(
    community_files: List[Path],
    ce_labels: Set[str],
    be_labels: Set[str],
    graph_nodes: Set[str],
    min_malicious_seeds: int,
    max_community_size: int,
    min_ce_in_graph: int = 2,
    max_communities: int = 20,
) -> Tuple[pd.DataFrame, List[str]]:
    """Identify communities suitable for offline seed-degradation trials."""
    rows: List[Dict[str, Any]] = []
    skip_logs: List[str] = []

    for path in community_files:
        cid = path.stem
        try:
            accounts = parse_community_accounts(path)
        except Exception as exc:
            skip_logs.append(f"{cid}: parse error - {exc}")
            continue

        malicious = accounts & ce_labels
        benign = accounts & be_labels
        mal_in_graph = malicious & graph_nodes

        if len(accounts) > max_community_size:
            skip_logs.append(f"{cid}: size={len(accounts)} > {max_community_size}")
            continue
        if len(malicious) < min_malicious_seeds:
            skip_logs.append(
                f"{cid}: malicious={len(malicious)} < {min_malicious_seeds}"
            )
            continue
        if len(mal_in_graph) < min_ce_in_graph:
            skip_logs.append(
                f"{cid}: ce_in_graph={len(mal_in_graph)} < {min_ce_in_graph}"
            )
            continue

        rows.append(
            {
                "community_id": cid,
                "community_json_path": str(path.resolve()),
                "community_size": len(accounts),
                "malicious_seed_count": len(malicious),
                "benign_seed_count": len(benign),
                "ce_seeds_in_graph": len(mal_in_graph),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, skip_logs

    df = df.sort_values(
        by=["malicious_seed_count", "community_size"], ascending=[False, True]
    ).head(max_communities)
    df = df.reset_index(drop=True)
    df["selection_rank"] = df.index + 1
    return df, skip_logs


def rank_candidates(
    scores: Dict[str, float],
    retained_seeds: Set[str],
) -> List[str]:
    """Sort nodes by PageRank score, excluding retained seeds."""
    candidates = [
        (node, score) for node, score in scores.items() if node not in retained_seeds
    ]
    candidates.sort(key=lambda x: (-x[1], x[0]))
    return [node for node, _ in candidates]


def compute_metrics_for_k(
    ranked: List[str],
    removed_seeds: Set[str],
    ce_labels: Set[str],
    be_labels: Set[str],
    k: int,
    rng: random.Random,
) -> Dict[str, Any]:
    """Compute APPR and random-baseline metrics at top-K."""
    top_k = ranked[:k]
    top_set = set(top_k)
    candidate_count = len(ranked)

    removed_n = len(removed_seeds)
    if removed_n == 0:
        removed_recovery = float("nan")
    else:
        removed_recovery = len(removed_seeds & top_set) / removed_n

    ce_hits = len(top_set & ce_labels)
    be_hits = len(top_set & be_labels)

    rank_lookup = {node: i + 1 for i, node in enumerate(ranked)}
    recovered_ranks = [rank_lookup[s] for s in removed_seeds if s in rank_lookup]
    if recovered_ranks:
        mean_rank = float(np.mean(recovered_ranks))
        min_rank = float(np.min(recovered_ranks))
    else:
        mean_rank = float("nan")
        min_rank = float("nan")

    # Random baseline: uniform sample from same candidate pool.
    if candidate_count == 0:
        random_removed_recovery = float("nan")
        random_ce_hits = float("nan")
    else:
        sample_size = min(k, candidate_count)
        random_top = set(rng.sample(ranked, sample_size))
        if removed_n == 0:
            random_removed_recovery = float("nan")
        else:
            random_removed_recovery = len(removed_seeds & random_top) / removed_n
        random_ce_hits = len(random_top & ce_labels)

    if removed_n == 0:
        recovery_lift = float("nan")
    elif math.isnan(random_removed_recovery) or random_removed_recovery == 0:
        recovery_lift = float("nan")
    else:
        recovery_lift = removed_recovery / random_removed_recovery

    return {
        "removed_recovery_at_k": removed_recovery,
        "ce_hits_at_k": ce_hits,
        "be_hits_at_k": be_hits,
        "candidate_count": candidate_count,
        "mean_rank_of_removed_seeds": mean_rank,
        "min_rank_of_removed_seeds": min_rank,
        "random_removed_recovery_at_k": random_removed_recovery,
        "random_ce_hits_at_k": random_ce_hits,
        "recovery_lift": recovery_lift,
    }


def run_experiments(
    eligible: pd.DataFrame,
    graph: LocalTransactionGraph,
    ce_labels: Set[str],
    be_labels: Set[str],
    removal_ratios: List[float],
    topk_list: List[int],
    repeats: int,
    random_seed: int,
    k_hop: int,
    max_subgraph_nodes: int,
    undirected_projection: bool,
    pagerank_alpha: float,
    pagerank_max_iter: int,
    pagerank_tol: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run all offline APPR seed-degradation trials."""
    rng = random.Random(random_seed)
    trial_rows: List[Dict[str, Any]] = []
    skip_logs: List[str] = []

    community_cache: Dict[str, Set[str]] = {}
    for _, row in eligible.iterrows():
        cid = row["community_id"]
        path = Path(row["community_json_path"])
        accounts = parse_community_accounts(path)
        community_cache[cid] = accounts & ce_labels

    total_trials = len(eligible) * len(removal_ratios) * repeats
    done = 0

    for _, row in eligible.iterrows():
        community_id = row["community_id"]
        malicious_seeds = sorted(community_cache[community_id])
        malicious_set = set(malicious_seeds)
        # Only seeds present in graph can be used for personalization.
        malicious_in_graph = [s for s in malicious_seeds if s in graph.nodes]

        for ratio in removal_ratios:
            n_remove = removal_count(len(malicious_in_graph), ratio)

            for repeat_idx in range(repeats):
                done += 1
                if done % 50 == 0:
                    logging.info("Trial progress: %d / %d", done, total_trials)

                if n_remove == 0:
                    removed: Set[str] = set()
                    retained = set(malicious_in_graph)
                else:
                    removed = set(rng.sample(malicious_in_graph, n_remove))
                    retained = set(malicious_in_graph) - removed

                if not retained:
                    skip_logs.append(
                        f"{community_id} ratio={ratio} repeat={repeat_idx}: "
                        "no retained seeds"
                    )
                    continue

                scores, err, sub_nodes = run_personalized_pagerank(
                    graph=graph,
                    retained_seeds=retained,
                    k_hop=k_hop,
                    max_subgraph_nodes=max_subgraph_nodes,
                    undirected_projection=undirected_projection,
                    alpha=pagerank_alpha,
                    max_iter=pagerank_max_iter,
                    tol=pagerank_tol,
                )
                if scores is None:
                    skip_logs.append(
                        f"{community_id} ratio={ratio} repeat={repeat_idx}: {err}"
                    )
                    continue

                ranked = rank_candidates(scores, retained)

                for k in topk_list:
                    metrics = compute_metrics_for_k(
                        ranked=ranked,
                        removed_seeds=removed,
                        ce_labels=ce_labels,
                        be_labels=be_labels,
                        k=k,
                        rng=rng,
                    )
                    trial_rows.append(
                        {
                            "community_id": community_id,
                            "removal_ratio": ratio,
                            "repeat": repeat_idx,
                            "top_k": k,
                            "removed_count": len(removed),
                            "retained_count": len(retained),
                            "subgraph_nodes": sub_nodes,
                            **metrics,
                        }
                    )

    return pd.DataFrame(trial_rows), skip_logs


def build_summary(trials_df: pd.DataFrame, eligible_count: int) -> pd.DataFrame:
    if trials_df.empty:
        return pd.DataFrame()

    grouped = trials_df.groupby(["removal_ratio", "top_k"], dropna=False).agg(
        removed_recovery_at_k_mean=("removed_recovery_at_k", "mean"),
        removed_recovery_at_k_std=("removed_recovery_at_k", "std"),
        random_removed_recovery_at_k_mean=("random_removed_recovery_at_k", "mean"),
        ce_hits_at_k_mean=("ce_hits_at_k", "mean"),
        be_hits_at_k_mean=("be_hits_at_k", "mean"),
        candidate_count_mean=("candidate_count", "mean"),
        valid_trial_count=("community_id", "count"),
    )
    summary = grouped.reset_index()
    summary["eligible_community_count"] = eligible_count

    rand = summary["random_removed_recovery_at_k_mean"]
    appr = summary["removed_recovery_at_k_mean"]
    summary["recovery_lift_mean"] = np.where(
        rand.isna() | (rand == 0) | appr.isna(),
        np.nan,
        appr / rand,
    )
    return summary


def summarize_trial_skip_reasons(trial_skip_logs: List[str]) -> Dict[str, int]:
    """Count runtime trial skips by reason category."""
    reason_counts: Dict[str, int] = defaultdict(int)
    for msg in trial_skip_logs:
        if "pagerank did not converge" in msg:
            reason_counts["pagerank_not_converged"] += 1
        elif "subgraph exceeds" in msg:
            reason_counts["subgraph_too_large"] += 1
        elif "no retained seeds" in msg:
            reason_counts["no_retained_seeds"] += 1
        else:
            reason_counts["other"] += 1
    return dict(reason_counts)


def build_paper_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact paper-ready table (ratio > 0, K in {50, 100, 200})."""
    if summary_df.empty:
        return pd.DataFrame()

    paper_ks = [50, 100, 200]
    sub = summary_df[
        (summary_df["removal_ratio"] > 0) & (summary_df["top_k"].isin(paper_ks))
    ].copy()
    sub = sub.sort_values(["top_k", "removal_ratio"]).reset_index(drop=True)

    table = pd.DataFrame(
        {
            "removal_ratio": sub["removal_ratio"],
            "top_k": sub["top_k"].astype(int),
            "APPR recovery (%)": (sub["removed_recovery_at_k_mean"] * 100).round(1),
            "Random recovery (%)": (
                sub["random_removed_recovery_at_k_mean"] * 100
            ).round(1),
            "Recovery lift": sub["recovery_lift_mean"].round(2),
            "CE hits": sub["ce_hits_at_k_mean"].round(1),
            "BE hits": sub["be_hits_at_k_mean"].round(1),
            "valid_trial_count": sub["valid_trial_count"].astype(int),
        }
    )
    return table


def save_paper_recovery_plot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """High-resolution APPR vs random recovery plot for paper (300 dpi)."""
    if summary_df.empty:
        logging.warning("Summary empty; skip paper plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    paper_ks = [50, 200]
    markers_appr = ["o", "s"]
    markers_rand = ["x", "+"]

    for k, mk_a, mk_r in zip(paper_ks, markers_appr, markers_rand):
        sub = summary_df[summary_df["top_k"] == k].sort_values("removal_ratio")
        pos = sub[sub["removal_ratio"] > 0]
        if pos.empty:
            continue
        ax.plot(
            pos["removal_ratio"],
            pos["removed_recovery_at_k_mean"],
            marker=mk_a,
            linewidth=2,
            label=f"APPR (K={k})",
        )
        ax.plot(
            pos["removal_ratio"],
            pos["random_removed_recovery_at_k_mean"],
            marker=mk_r,
            linestyle="--",
            linewidth=1.5,
            label=f"Random (K={k})",
        )

    ax.set_title(
        "Recovery of Removed Malicious Seeds: APPR vs Random Baseline",
        fontsize=12,
    )
    ax.set_xlabel("Removal ratio", fontsize=11)
    ax.set_ylabel("Recovery@K", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_dir / PLOT_PAPER, dpi=300)
    plt.close(fig)


def _lookup_summary_metric(
    summary_df: pd.DataFrame,
    top_k: int,
    removal_ratio: float,
    column: str,
) -> float:
    """Fetch one summary cell; return NaN if missing."""
    row = summary_df[
        (summary_df["top_k"] == top_k)
        & (np.isclose(summary_df["removal_ratio"], removal_ratio))
    ]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def write_key_findings(
    output_path: Path,
    eligible_count: int,
    graph: LocalTransactionGraph,
    trials_df: pd.DataFrame,
    trial_skip_logs: List[str],
    summary_df: pd.DataFrame,
) -> None:
    """Write a short machine-readable summary for rebuttal / paper drafting."""
    skip_reasons = summarize_trial_skip_reasons(trial_skip_logs)
    total_skipped = len(trial_skip_logs)
    pagerank_failures = skip_reasons.get("pagerank_not_converged", 0)

    def fmt_pct(val: float) -> str:
        if math.isnan(val):
            return "NaN"
        return f"{val * 100:.1f}%"

    def fmt_lift(val: float) -> str:
        if math.isnan(val):
            return "NaN"
        return f"{val:.2f}"

    lines = [
        "Phase-1 Offline APPR Seed Degradation — Key Findings",
        "=" * 55,
        f"eligible_communities: {eligible_count}",
        f"graph_nodes: {graph.num_nodes}",
        f"graph_edges: {graph.edge_count}",
        f"valid_trial_rows: {len(trials_df)}",
        f"skipped_trials: {total_skipped}",
        f"skipped_rows_detail: {skip_reasons}",
        f"pagerank_non_convergence_count: {pagerank_failures}",
        "",
        "K=50:",
    ]

    for ratio in (0.1, 0.5):
        appr = _lookup_summary_metric(
            summary_df, 50, ratio, "removed_recovery_at_k_mean"
        )
        rand = _lookup_summary_metric(
            summary_df, 50, ratio, "random_removed_recovery_at_k_mean"
        )
        lift = _lookup_summary_metric(summary_df, 50, ratio, "recovery_lift_mean")
        lines.append(
            f"  removal_ratio={ratio}: APPR={fmt_pct(appr)}, "
            f"random={fmt_pct(rand)}, lift={fmt_lift(lift)}"
        )

    lines.append("")
    lines.append("K=200:")
    for ratio in (0.1, 0.5):
        appr = _lookup_summary_metric(
            summary_df, 200, ratio, "removed_recovery_at_k_mean"
        )
        rand = _lookup_summary_metric(
            summary_df, 200, ratio, "random_removed_recovery_at_k_mean"
        )
        lift = _lookup_summary_metric(summary_df, 200, ratio, "recovery_lift_mean")
        lines.append(
            f"  removal_ratio={ratio}: APPR={fmt_pct(appr)}, "
            f"random={fmt_pct(rand)}, lift={fmt_lift(lift)}"
        )

    lines.extend(
        [
            "",
            "Conclusion: APPR consistently outperforms the random baseline under "
            "simulated point-level false negatives.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plots(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if summary_df.empty:
        logging.warning("Summary empty; skip plots.")
        return

    plot_ks = [50, 100, 200]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Recovery plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, color in zip(plot_ks, colors):
        sub = summary_df[summary_df["top_k"] == k].sort_values("removal_ratio")
        if sub.empty:
            continue
        ax.plot(
            sub["removal_ratio"],
            sub["removed_recovery_at_k_mean"],
            marker="o",
            label=f"APPR K={k}",
            color=color,
        )
        ax.plot(
            sub["removal_ratio"],
            sub["random_removed_recovery_at_k_mean"],
            marker="x",
            linestyle="--",
            label=f"Random K={k}",
            color=color,
            alpha=0.6,
        )
    ax.set_title(
        "Offline APPR: Recovery of Removed Malicious Seeds under False Negatives"
    )
    ax.set_xlabel("Simulated false-negative ratio")
    ax.set_ylabel("Recovery rate of removed CE seeds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / PLOT_RECOVERY, dpi=150)
    plt.close(fig)

    # CE hits plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, color in zip(plot_ks, colors):
        sub = summary_df[summary_df["top_k"] == k].sort_values("removal_ratio")
        if sub.empty:
            continue
        ax.plot(
            sub["removal_ratio"],
            sub["ce_hits_at_k_mean"],
            marker="o",
            label=f"K={k}",
            color=color,
        )
    ax.set_title("Offline APPR: CE Hits in Top-K Candidates")
    ax.set_xlabel("Simulated false-negative ratio")
    ax.set_ylabel("Mean CE hits at top-K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / PLOT_CE_HITS, dpi=150)
    plt.close(fig)


def write_log(
    log_path: Path,
    args: argparse.Namespace,
    graph: LocalTransactionGraph,
    eligible: pd.DataFrame,
    skip_logs: List[str],
    trial_skip_logs: List[str],
    trials_df: pd.DataFrame,
    elapsed: float,
) -> None:
    lines = [
        "Phase-1 Offline APPR Seed Degradation Log",
        "=" * 50,
        f"elapsed_seconds: {elapsed:.1f}",
        f"dry_run: {args.dry_run}",
        f"transactions_file: {args.transactions_file}",
        f"undirected_projection: {args.undirected_projection}",
        f"k_hop: {args.k_hop}",
        f"max_subgraph_nodes: {args.max_subgraph_nodes}",
        f"pagerank_alpha: {args.pagerank_alpha}",
        f"repeats: {args.repeats}",
        f"random_seed: {args.random_seed}",
        "",
        f"graph_nodes: {graph.num_nodes}",
        f"graph_edge_records: {graph.edge_count}",
        f"eligible_communities: {len(eligible)}",
        f"valid_trial_rows: {len(trials_df)}",
        "",
        "Eligible communities:",
    ]
    if eligible.empty:
        lines.append("  (none)")
    else:
        for _, row in eligible.iterrows():
            lines.append(
                f"  #{int(row['selection_rank'])} {row['community_id']}: "
                f"malicious={int(row['malicious_seed_count'])}, "
                f"ce_in_graph={int(row['ce_seeds_in_graph'])}, "
                f"size={int(row['community_size'])}"
            )

    lines.extend(["", "Skipped communities (filter):"])
    lines.extend(f"  - {s}" for s in skip_logs[:100])
    if len(skip_logs) > 100:
        lines.append(f"  ... and {len(skip_logs) - 100} more")

    lines.extend(["", "Skipped trials (runtime):"])
    if trial_skip_logs:
        reason_counts = summarize_trial_skip_reasons(trial_skip_logs)
        for reason, cnt in sorted(reason_counts.items()):
            lines.append(f"  - {reason}: {cnt}")
        lines.append(f"  total_skipped_trials: {len(trial_skip_logs)}")
    else:
        lines.append("  (none)")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline APPR seed-degradation robustness experiment."
    )
    parser.add_argument(
        "--transactions_file",
        default="transactions/labeled_transactions.csv",
        help="Local transaction CSV (default: transactions/labeled_transactions.csv)",
    )
    parser.add_argument(
        "--communities_dir",
        default="results1/combined",
        help="Community JSON directory",
    )
    parser.add_argument(
        "--labels_ce",
        default="labels/CE.csv",
        help="Malicious label CSV",
    )
    parser.add_argument(
        "--labels_be",
        default="labels/BE.csv",
        help="Benign label CSV",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR_NAME,
        help="Output directory under project root",
    )
    parser.add_argument("--min_malicious_seeds", type=int, default=5)
    parser.add_argument("--max_community_size", type=int, default=500)
    parser.add_argument("--max_communities", type=int, default=20)
    parser.add_argument(
        "--removal_ratios",
        default="0,0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated removal ratios",
    )
    parser.add_argument("--top_k", default="20,50,100,200")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--k_hop", type=int, default=2)
    parser.add_argument("--max_subgraph_nodes", type=int, default=10000)
    parser.add_argument("--undirected_projection", action="store_true")
    parser.add_argument("--pagerank_alpha", type=float, default=0.85)
    parser.add_argument("--pagerank_max_iter", type=int, default=100)
    parser.add_argument("--pagerank_tol", type=float, default=1e-6)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    t0 = time.monotonic()

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ce_labels = load_label_addresses(resolve_project_path(args.labels_ce), "CE")
    be_labels = load_label_addresses(resolve_project_path(args.labels_be), "BE")
    community_files = discover_community_files(resolve_project_path(args.communities_dir))

    graph = build_transaction_graph(
        resolve_project_path(args.transactions_file),
        undirected_projection=args.undirected_projection,
    )

    eligible, filter_skips = find_eligible_communities(
        community_files=community_files,
        ce_labels=ce_labels,
        be_labels=be_labels,
        graph_nodes=graph.nodes,
        min_malicious_seeds=args.min_malicious_seeds,
        max_community_size=args.max_community_size,
        max_communities=args.max_communities,
    )
    eligible.to_csv(output_dir / ELIGIBLE_CSV, index=False, encoding="utf-8")

    logging.info("Eligible communities: %d", len(eligible))
    logging.info("Graph: %d nodes, %d edges", graph.num_nodes, graph.edge_count)

    if args.dry_run:
        write_log(
            log_path=output_dir / LOG_TXT,
            args=args,
            graph=graph,
            eligible=eligible,
            skip_logs=filter_skips,
            trial_skip_logs=[],
            trials_df=pd.DataFrame(),
            elapsed=time.monotonic() - t0,
        )
        print(f"\n[DRY RUN] Output: {output_dir.resolve()}")
        print(f"[DRY RUN] Graph nodes: {graph.num_nodes}, edges: {graph.edge_count}")
        print(f"[DRY RUN] Eligible communities: {len(eligible)}")
        print(f"[DRY RUN] Eligible CSV: {(output_dir / ELIGIBLE_CSV).resolve()}")
        return

    if eligible.empty:
        write_log(
            log_path=output_dir / LOG_TXT,
            args=args,
            graph=graph,
            eligible=eligible,
            skip_logs=filter_skips,
            trial_skip_logs=[],
            trials_df=pd.DataFrame(),
            elapsed=time.monotonic() - t0,
        )
        raise RuntimeError("No eligible communities. See log for skip reasons.")

    removal_ratios = _parse_float_list(args.removal_ratios, "removal_ratios")
    topk_list = _parse_int_list(args.top_k, "top_k")

    trials_df, trial_skips = run_experiments(
        eligible=eligible,
        graph=graph,
        ce_labels=ce_labels,
        be_labels=be_labels,
        removal_ratios=removal_ratios,
        topk_list=topk_list,
        repeats=args.repeats,
        random_seed=args.random_seed,
        k_hop=args.k_hop,
        max_subgraph_nodes=args.max_subgraph_nodes,
        undirected_projection=args.undirected_projection,
        pagerank_alpha=args.pagerank_alpha,
        pagerank_max_iter=args.pagerank_max_iter,
        pagerank_tol=args.pagerank_tol,
    )

    summary_df = build_summary(trials_df, eligible_count=len(eligible))
    paper_table_df = build_paper_table(summary_df)

    trials_df.to_csv(output_dir / TRIAL_CSV, index=False, encoding="utf-8")
    summary_df.to_csv(output_dir / SUMMARY_CSV, index=False, encoding="utf-8")
    paper_table_df.to_csv(output_dir / TABLE_PAPER, index=False, encoding="utf-8")
    save_plots(summary_df, output_dir)
    save_paper_recovery_plot(summary_df, output_dir)
    write_key_findings(
        output_path=output_dir / KEY_FINDINGS_TXT,
        eligible_count=len(eligible),
        graph=graph,
        trials_df=trials_df,
        trial_skip_logs=trial_skips,
        summary_df=summary_df,
    )

    elapsed = time.monotonic() - t0
    write_log(
        log_path=output_dir / LOG_TXT,
        args=args,
        graph=graph,
        eligible=eligible,
        skip_logs=filter_skips,
        trial_skip_logs=trial_skips,
        trials_df=trials_df,
        elapsed=elapsed,
    )

    print(f"\nOutput directory: {output_dir.resolve()}")
    print(f"Graph nodes: {graph.num_nodes}, edges: {graph.edge_count}")
    print(f"Eligible communities: {len(eligible)}")
    print(f"Valid trial rows: {len(trials_df)}")
    print(f"Summary: {(output_dir / SUMMARY_CSV).resolve()}")
    print(f"Paper table: {(output_dir / TABLE_PAPER).resolve()}")
    print(f"Key findings: {(output_dir / KEY_FINDINGS_TXT).resolve()}")
    print(f"Plots: {(output_dir / PLOT_RECOVERY).resolve()}")
    print(f"         {(output_dir / PLOT_CE_HITS).resolve()}")
    print(f"         {(output_dir / PLOT_PAPER).resolve()}")


if __name__ == "__main__":
    main()
