"""Metric helpers for clean main experiments."""

from __future__ import annotations

import random
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reviewer2_major1_ablation"))
from ablation_utils import (  # noqa: E402
    BE_PATH,
    CE_PATH,
    build_transaction_graph,
    load_community_members,
    load_label_addresses as _load_label_addresses,
    normalize_address,
    _read_log_text,
)


def load_label_addresses() -> Tuple[Set[str], Set[str]]:
    return _load_label_addresses(CE_PATH), _load_label_addresses(BE_PATH)

BE_RELAXED = 0.15


def total_ce_in_graph(graph_nodes: Set[str], ce: Set[str]) -> int:
    return len(ce & graph_nodes)


def labeled_account_metrics(
    predicted: Set[str],
    ce: Set[str],
    be: Set[str],
    graph_nodes: Set[str],
    scores: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    ce_h = len(predicted & ce)
    be_h = len(predicted & be)
    pred_n = len(predicted)
    unknown = pred_n - ce_h - be_h
    total_ce = total_ce_in_graph(graph_nodes, ce)
    lp = ce_h / (ce_h + be_h) if ce_h + be_h else 0.0
    lr = ce_h / total_ce if total_ce else 0.0
    lf1 = 2 * lp * lr / (lp + lr) if lp + lr else 0.0
    strict_tp = ce_h + be_h
    if scores:
        extra_be = sum(
            1 for a in be & graph_nodes
            if a not in predicted and scores.get(a, 0) >= BE_RELAXED
        )
        strict_tp += extra_be
    strict_p = strict_tp / (strict_tp + unknown) if strict_tp + unknown else 0.0
    return {
        "strict_precision": strict_p,
        "labeled_precision": lp,
        "labeled_recall": lr,
        "labeled_f1": lf1,
        "predicted_suspicious": pred_n,
        "ce_hits": ce_h,
        "be_hits": be_h,
        "unknown_predictions": unknown,
        "unknown_rate": unknown / pred_n if pred_n else 0.0,
    }


def compute_auc(scores: Dict[str, float], ce: Set[str], be: Set[str], graph_nodes: Set[str]) -> float:
    labeled = (ce | be) & graph_nodes
    y_true, y_scores = [], []
    for a in labeled:
        if a in scores:
            y_true.append(1)
            y_scores.append(scores[a])
    neg = [n for n in graph_nodes if n not in labeled and n in scores]
    random.seed(42)
    k = min(len(neg), len(labeled))
    for a in random.sample(neg, k) if k else []:
        y_true.append(0)
        y_scores.append(scores[a])
    return float(roc_auc_score(y_true, y_scores)) if len(set(y_true)) > 1 else 0.0


def parse_log_strict_metrics(log_path: Path) -> Dict[str, float]:
    raw = log_path.read_bytes() if log_path.is_file() else b""
    text = _read_log_text(log_path)
    if "精确率" not in text and "10.80" in raw.decode("gbk", errors="replace"):
        text = raw.decode("gbk", errors="replace")
    out = {}
    for key, pat in [
        ("precision", r"精确率:\s*([\d.]+)%"),
        ("recall", r"召回率:\s*([\d.]+)%"),
        ("f1", r"F1分数:\s*([\d.]+)%"),
        ("auc", r"AUC值:\s*([\d.]+)"),
    ]:
        m = re.search(pat, text)
        if m:
            v = float(m.group(1))
            out[key] = v / 100 if key != "auc" else v
    return out


def group_metrics_from_communities(
    communities: List[Set[str]],
    ce: Set[str],
    be: Set[str],
    graph_nodes: Set[str],
) -> Dict[str, float]:
    if not communities:
        return {"ac": 0.0, "nmi": 0.0, "ari": 0.0, "num_groups": 0, "max_size_ratio": 0.0,
                "ce_coverage": 0.0, "labeled_precision": 0.0}
    ce_g = ce & graph_nodes
    sizes = [len(c) for c in communities]
    total = sum(sizes)
    covered, ce_cnt, be_cnt = set(), 0, 0
    conductances = []
    ug = build_transaction_graph().to_undirected()
    for comm in communities:
        cc, cb = comm & ce_g, comm & be
        covered |= cc
        ce_cnt += len(cc)
        be_cnt += len(cb)
        sub = ug.subgraph(comm)
        if sub.number_of_edges():
            cut = vol = 0.0
            for u in comm:
                for v in sub.neighbors(u):
                    w = sub[u][v].get("weight", 1.0)
                    vol += w
                    if v not in comm:
                        cut += w
            if vol:
                conductances.append(cut / vol)
    # label-based partition for NMI/ARI on labeled nodes
    labeled_nodes = list((ce_g | be) & set().union(*communities))
    true_labels, pred_labels = [], []
    node_to_pred = {}
    for i, comm in enumerate(communities):
        for n in comm:
            node_to_pred[n] = i
    for n in labeled_nodes:
        true_labels.append(1 if n in ce_g else 0)
        pred_labels.append(node_to_pred.get(n, -1))
    nmi = normalized_mutual_info_score(true_labels, pred_labels) if len(set(pred_labels)) > 1 else 0.0
    ari = adjusted_rand_score(true_labels, pred_labels) if len(set(pred_labels)) > 1 else 0.0
    return {
        "ac": statistics.mean(conductances) if conductances else 0.0,
        "nmi": nmi,
        "ari": ari,
        "num_groups": len(communities),
        "max_size_ratio": max(sizes) / total if total else 0.0,
        "ce_coverage": len(covered) / len(ce_g) if ce_g else 0.0,
        "labeled_precision": ce_cnt / (ce_cnt + be_cnt) if ce_cnt + be_cnt else 0.0,
    }
