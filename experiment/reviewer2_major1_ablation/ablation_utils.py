"""Shared utilities for Reviewer #2 Major Concern 1 ablation experiments."""

from __future__ import annotations

import csv
import glob
import logging
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results_ablation" / "reviewer2_major1"
TABLES_DIR = RESULTS_ROOT / "tables"
LOGS_DIR = RESULTS_ROOT / "logs"
TX_PATH = PROJECT_ROOT / "transactions" / "labeled_transactions.csv"
CE_PATH = PROJECT_ROOT / "labels" / "CE.csv"
BE_PATH = PROJECT_ROOT / "labels" / "BE.csv"

FROM_ALIASES = ("from", "sender", "source_address", "src")
TO_ALIASES = ("to", "recipient", "target_address", "dst")


def normalize_address(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if not text.startswith("0x"):
        text = f"0x{text}"
    if len(text) >= 42:
        text = text[:42]
    if len(text) == 42 and text.startswith("0x"):
        return text
    return None


def load_label_addresses(path: Path) -> Set[str]:
    if not path.is_file():
        return set()
    df = pd.read_csv(path, encoding="utf-8")
    addr_cols = [c for c in df.columns if c.lower() == "address"]
    if not addr_cols:
        return set()
    addrs = set()
    for raw in df[addr_cols[0]]:
        addr = normalize_address(raw)
        if addr:
            addrs.add(addr)
    return addrs


def pick_column(columns: List[str], aliases: Tuple[str, ...]) -> str:
    lower_map = {c.lower(): c for c in columns}
    for name in aliases:
        if name in lower_map:
            return lower_map[name]
    raise ValueError(f"Missing column aliases {aliases} in {columns}")


def build_transaction_graph(tx_path: Path = TX_PATH) -> nx.DiGraph:
    header = pd.read_csv(tx_path, nrows=0, encoding="utf-8")
    columns = list(header.columns)
    from_col = pick_column(columns, FROM_ALIASES)
    to_col = pick_column(columns, TO_ALIASES)
    g = nx.DiGraph()
    for chunk in pd.read_csv(tx_path, usecols=[from_col, to_col], chunksize=200_000, encoding="utf-8"):
        for _, row in chunk.iterrows():
            src = normalize_address(row[from_col])
            dst = normalize_address(row[to_col])
            if src and dst and src != dst:
                g.add_edge(src, dst)
    return g


def _read_log_text(log_path: Path) -> str:
    if not log_path.is_file():
        return ""
    raw = log_path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def parse_log_metrics(log_path: Path) -> Dict[str, Optional[float]]:
    text = _read_log_text(log_path)
    patterns = {
        "precision": r"精确率:\s*([\d.]+)%",
        "recall": r"召回率:\s*([\d.]+)%",
        "f1": r"F1分数:\s*([\d.]+)%",
        "auc": r"AUC值:\s*([\d.]+)",
    }
    out: Dict[str, Optional[float]] = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            out[key] = val / 100.0 if key != "auc" else val
        else:
            out[key] = None
    return out


def count_csv_rows(path: Path, skip_header: bool = True) -> int:
    if not path.is_file():
        return 0
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader, None)
        return sum(1 for _ in reader)


def find_suspicious_accounts_csv(out_dir: Path) -> Optional[Path]:
    matches = sorted(out_dir.glob("*_suspicious_accounts.csv"))
    return matches[0] if matches else None


def load_predicted_suspicious(out_dir: Path) -> Set[str]:
    path = find_suspicious_accounts_csv(out_dir)
    if not path:
        return set()
    addrs = set()
    df = pd.read_csv(path, encoding="utf-8")
    node_col = "Node" if "Node" in df.columns else df.columns[0]
    for raw in df[node_col]:
        addr = normalize_address(raw)
        if addr:
            addrs.add(addr)
    return addrs


def load_community_members(community_dir: Path, filename_prefix: str = "labeled_transactions") -> List[Set[str]]:
    communities: List[Set[str]] = []
    pattern = str(community_dir / f"{filename_prefix}_community_*.csv")
    for path in sorted(glob.glob(pattern), key=lambda p: int(Path(p).stem.split("_")[-1])):
        df = pd.read_csv(path, encoding="utf-8")
        node_col = "Node" if "Node" in df.columns else df.columns[0]
        members = set()
        for raw in df[node_col]:
            addr = normalize_address(raw)
            if addr:
                members.add(addr)
        if members:
            communities.append(members)
    return communities


def compute_community_metrics(
    communities: List[Set[str]],
    ce_addrs: Set[str],
    be_addrs: Set[str],
    graph_nodes: Optional[Set[str]] = None,
) -> Dict[str, object]:
    if graph_nodes is not None:
        ce_in_graph = ce_addrs & graph_nodes
        be_in_graph = be_addrs & graph_nodes
    else:
        ce_in_graph = ce_addrs
        be_in_graph = be_addrs

    if not communities:
        return {
            "num_communities": 0,
            "avg_community_size": 0.0,
            "max_community_size": 0,
            "ce_coverage": 0.0,
            "labeled_precision": "NA",
            "avg_conductance": "NA",
            "nmi": "NA",
            "ari": "NA",
        }

    sizes = [len(c) for c in communities]
    covered_ce: Set[str] = set()
    ce_in_comm = 0
    be_in_comm = 0
    for comm in communities:
        comm_ce = comm & ce_in_graph
        comm_be = comm & be_in_graph
        covered_ce.update(comm_ce)
        ce_in_comm += len(comm_ce)
        be_in_comm += len(comm_be)

    ce_coverage = len(covered_ce) / len(ce_in_graph) if ce_in_graph else 0.0
    labeled_total = ce_in_comm + be_in_comm
    labeled_precision = ce_in_comm / labeled_total if labeled_total > 0 else "NA"

    return {
        "num_communities": len(communities),
        "avg_community_size": sum(sizes) / len(sizes),
        "max_community_size": max(sizes),
        "ce_coverage": ce_coverage,
        "labeled_precision": labeled_precision,
        "avg_conductance": "NA",
        "nmi": "NA",
        "ari": "NA",
    }


def parse_pairwise_nmi_ari(log_path: Path, algo: str) -> Tuple[str, str]:
    """Extract average NMI/ARI involving algo from log; return NA if unavailable."""
    text = _read_log_text(log_path)
    if not text:
        return "NA", "NA"
    nmi_vals: List[float] = []
    ari_vals: List[float] = []
    blocks = re.findall(rf"(\w+) vs (\w+):[\s\S]*?NMI\):\s*([\d.]+)[\s\S]*?ARI\):\s*([\d.]+)", text)
    for a1, a2, nmi_s, ari_s in blocks:
        if algo in (a1, a2):
            nmi_vals.append(float(nmi_s))
            ari_vals.append(float(ari_s))
    if not nmi_vals:
        return "NA", "NA"
    return f"{sum(nmi_vals)/len(nmi_vals):.4f}", f"{sum(ari_vals)/len(ari_vals):.4f}"


def write_summary_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logging.info("Wrote %s", path)
