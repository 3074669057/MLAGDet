#!/usr/bin/env python3
"""
Phase-2 point-level fusion weight (omega) sensitivity experiment.

Reuses existing rule_score and model_score (from prior outputs or by calling
SuspiciousAccountDetector scoring functions) and sweeps omega in
    S(v) = omega * S_ML(v) + (1 - omega) * S_rule(v)
without retraining models, running APPR, or community discovery.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "phase2" / "alpha_sensitivity"
LOG_PATH = OUTPUT_DIR / "logs" / "phase2_alpha_sensitivity_log.txt"
RESULTS_CSV = OUTPUT_DIR / "phase2_alpha_sensitivity_results.csv"
PLOT_PATH = OUTPUT_DIR / "figures" / "phase2_alpha_sensitivity_plot.png"
SCORE_CACHE = OUTPUT_DIR / "intermediate" / "point_level_scores.csv"

OMEGA_VALUES = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)
RANDOM_STATE = 0

SCORE_FILE_GLOBS = [
    "results1/**/*detailed*.csv",
    "results1/**/*suspicious*.csv",
    "results_final11/**/*detailed*.csv",
    "results_final11/**/*suspicious*.csv",
    "results*/**/*detailed*.csv",
    "results*/**/*suspicious*.csv",
]

ADDRESS_ALIASES = ("address", "account", "node")
RULE_SCORE_ALIASES = ("rule_score", "rulescore")
MODEL_SCORE_ALIASES = ("model_score", "modelscore")
FINAL_SCORE_ALIASES = ("final_score", "suspiciousscore", "suspiciousprobability", "score")

DEFAULT_TXS_FILE = PROJECT_ROOT / "transactions" / "labeled_transactions.csv"


def setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "intermediate").mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("phase2_alpha_sensitivity")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def normalize_address(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if not text.startswith("0x"):
        text = f"0x{text}"
    if len(text) >= 42:
        text = text[:42]
    return text


def _pick_column(columns: List[str], aliases: Tuple[str, ...]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for name in aliases:
        if name in lower_map:
            return lower_map[name]
    return None


def load_ground_truth() -> pd.DataFrame:
    ce_path = PROJECT_ROOT / "labels" / "CE.csv"
    be_path = PROJECT_ROOT / "labels" / "BE.csv"
    ce = pd.read_csv(ce_path)
    be = pd.read_csv(be_path)

    ce_addr_col = _pick_column(list(ce.columns), ADDRESS_ALIASES)
    be_addr_col = _pick_column(list(be.columns), ADDRESS_ALIASES)
    if ce_addr_col is None or be_addr_col is None:
        raise ValueError("CE.csv or BE.csv is missing an address column.")

    ce_df = pd.DataFrame(
        {
            "address": ce[ce_addr_col].map(normalize_address),
            "label": 1,
        }
    ).dropna(subset=["address"])
    be_df = pd.DataFrame(
        {
            "address": be[be_addr_col].map(normalize_address),
            "label": 0,
        }
    ).dropna(subset=["address"])

    labels = pd.concat([ce_df, be_df], ignore_index=True)
    labels = labels.drop_duplicates(subset=["address"], keep="first")
    return labels


def discover_score_files() -> List[Path]:
    found: List[Path] = []
    seen = set()
    for pattern in SCORE_FILE_GLOBS:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                found.append(path)
    return sorted(found)


def read_score_file(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    addr_col = _pick_column(list(df.columns), ADDRESS_ALIASES)
    rule_col = _pick_column(list(df.columns), RULE_SCORE_ALIASES)
    model_col = _pick_column(list(df.columns), MODEL_SCORE_ALIASES)
    if addr_col is None or rule_col is None or model_col is None:
        return None

    out = pd.DataFrame(
        {
            "address": df[addr_col].map(normalize_address),
            "rule_score": pd.to_numeric(df[rule_col], errors="coerce"),
            "model_score": pd.to_numeric(df[model_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["address", "rule_score", "model_score"])
    out = out.drop_duplicates(subset=["address"], keep="first")
    return out if not out.empty else None


def try_load_existing_scores(
    labels: pd.DataFrame, logger: logging.Logger
) -> Tuple[Optional[pd.DataFrame], List[str], bool]:
    """Return (scores_df, searched_paths, loaded_from_file)."""
    searched = [str(p.relative_to(PROJECT_ROOT)) for p in discover_score_files()]
    labeled_addrs = set(labels["address"])
    best: Optional[pd.DataFrame] = None
    best_path: Optional[Path] = None
    best_overlap = 0

    for path in discover_score_files():
        scores = read_score_file(path)
        if scores is None:
            continue
        overlap = len(set(scores["address"]) & labeled_addrs)
        if overlap > best_overlap:
            best_overlap = overlap
            best = scores
            best_path = path

    if best is None or best_overlap == 0:
        return None, searched, False

    merged = labels.merge(best, on="address", how="inner")
    if merged.empty:
        return None, searched, False

    logger.info(
        "Loaded rule_score/model_score from existing file: %s (overlap with labeled accounts: %d)",
        best_path,
        len(merged),
    )
    return merged, searched, True


def compute_scores_with_detector(
    txs_file: Path, logger: logging.Logger
) -> pd.DataFrame:
    if not txs_file.exists():
        raise FileNotFoundError(f"Transaction file not found: {txs_file}")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    temp_out = OUTPUT_DIR / "intermediate" / "detector_scratch"
    temp_out.mkdir(parents=True, exist_ok=True)
    dummy_nodes = temp_out / "all_nodes.csv"
    with open(dummy_nodes, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node"])

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "phase2_alpha_sensitivity_experiment.py",
            "-x",
            str(txs_file),
            "-n",
            str(dummy_nodes),
            "-o",
            str(temp_out),
        ]
        from suspicious_account_detector import SuspiciousAccountDetector

        detector = SuspiciousAccountDetector()
        logger.info("Building transaction graph from %s", txs_file)
        graph = detector.load_graph(str(txs_file))
        logger.info("Extracting node features (%d nodes)...", graph.number_of_nodes())
        features = detector.extract_features(graph)
        logger.info("Running rule_based_detection...")
        rule_results = detector.rule_based_detection(features)
        logger.info("Running model_based_detection...")
        model_results = (
            detector.model_based_detection(features) if detector.models else {}
        )
        if not detector.models:
            raise RuntimeError("No pretrained models loaded; cannot compute model_score.")
    finally:
        sys.argv = old_argv

    rows = []
    for node, feat in features.items():
        addr = normalize_address(node)
        if addr is None:
            continue
        rule_score = rule_results.get(node, {}).get("probability")
        if rule_score is None:
            continue
        model_score = model_results.get(node, 0.0)
        rows.append(
            {
                "address": addr,
                "rule_score": float(rule_score),
                "model_score": float(model_score),
            }
        )
    return pd.DataFrame(rows)


def stratified_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=train_val["label"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def fuse_scores(
    rule_score: np.ndarray, model_score: np.ndarray, omega: float
) -> np.ndarray:
    return omega * model_score + (1.0 - omega) * rule_score


def search_best_threshold(
    y_true: np.ndarray, scores: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_f1 = -1.0
    best_metrics = {
        "val_precision": 0.0,
        "val_recall": 0.0,
        "val_f1": 0.0,
        "val_auc": 0.0,
    }

    for threshold in THRESHOLD_GRID:
        y_pred = (scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_true, scores)
            except ValueError:
                auc = float("nan")
            best_metrics = {
                "val_precision": prec,
                "val_recall": rec,
                "val_f1": f1,
                "val_auc": auc,
            }

    return best_threshold, best_metrics


def evaluate_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")
    return {
        "test_precision": precision_score(y_true, y_pred, zero_division=0),
        "test_recall": recall_score(y_true, y_pred, zero_division=0),
        "test_f1": f1_score(y_true, y_pred, zero_division=0),
        "test_auc": auc,
    }


def run_omega_grid(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    results = []
    y_val = val["label"].to_numpy()
    y_test = test["label"].to_numpy()
    rule_val = val["rule_score"].to_numpy()
    model_val = val["model_score"].to_numpy()
    rule_test = test["rule_score"].to_numpy()
    model_test = test["model_score"].to_numpy()

    for omega in OMEGA_VALUES:
        val_scores = fuse_scores(rule_val, model_val, omega)
        best_threshold, val_metrics = search_best_threshold(y_val, val_scores)
        test_scores = fuse_scores(rule_test, model_test, omega)
        test_metrics = evaluate_at_threshold(y_test, test_scores, best_threshold)

        row = {
            "omega": omega,
            "rule_weight": round(1.0 - omega, 1),
            "model_weight": omega,
            "best_threshold": best_threshold,
            **val_metrics,
            **test_metrics,
        }
        results.append(row)

    return pd.DataFrame(results)


def save_plot(results_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["omega"], results_df["test_precision"], marker="o", label="test_precision")
    ax.plot(results_df["omega"], results_df["test_recall"], marker="o", label="test_recall")
    ax.plot(results_df["omega"], results_df["test_f1"], marker="o", label="test_f1")
    ax.plot(results_df["omega"], results_df["test_auc"], marker="o", label="test_auc")
    ax.set_title("Sensitivity Analysis of Fusion Weight omega")
    ax.set_xlabel("omega")
    ax.set_ylabel("Score")
    ax.set_xticks(OMEGA_VALUES)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


def write_summary_log(
    logger: logging.Logger,
    *,
    input_paths: List[str],
    loaded_from_file: bool,
    recomputed: bool,
    labels: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    results_df: pd.DataFrame,
    elapsed: float,
) -> None:
    ce_count = int((labels["label"] == 1).sum())
    be_count = int((labels["label"] == 0).sum())

    logger.info("=== Phase-2 omega sensitivity experiment log ===")
    logger.info("Input file paths searched / used:")
    for p in input_paths:
        logger.info("  - %s", p)
    logger.info("Loaded rule_score/model_score from existing file: %s", loaded_from_file)
    logger.info("Recomputed rule_score/model_score via detector functions: %s", recomputed)
    logger.info("Total labeled evaluation samples: %d", len(labels))
    logger.info("CE (malicious, label=1) samples: %d", ce_count)
    logger.info("BE (benign, label=0) samples: %d", be_count)
    logger.info("Train samples: %d", len(train))
    logger.info("Validation samples: %d", len(val))
    logger.info("Test samples: %d", len(test))
    logger.info("Elapsed time: %.1f s", elapsed)

    def log_omega_row(omega: float, tag: str) -> None:
        row = results_df.loc[results_df["omega"] == omega].iloc[0]
        logger.info(
            "%s (omega=%.1f): test_precision=%.4f test_recall=%.4f test_f1=%.4f test_auc=%.4f best_threshold=%.2f",
            tag,
            omega,
            row["test_precision"],
            row["test_recall"],
            row["test_f1"],
            row["test_auc"],
            row["best_threshold"],
        )

    log_omega_row(0.0, "Rule-only")
    log_omega_row(1.0, "Model-only")
    log_omega_row(0.6, "Current paper setting")

    best_val_idx = results_df["val_f1"].idxmax()
    best_test_idx = results_df["test_f1"].idxmax()
    best_val_omega = float(results_df.loc[best_val_idx, "omega"])
    best_test_omega = float(results_df.loc[best_test_idx, "omega"])
    best_test_f1 = float(results_df.loc[best_test_idx, "test_f1"])

    logger.info("Best validation F1 omega: %.1f (val_f1=%.4f)", best_val_omega, results_df.loc[best_val_idx, "val_f1"])
    logger.info("Best test F1 omega: %.1f (test_f1=%.4f)", best_test_omega, best_test_f1)

    paper_row = results_df.loc[results_df["omega"] == 0.6].iloc[0]
    if best_test_omega != 0.6:
        delta = best_test_f1 - float(paper_row["test_f1"])
        logger.info(
            "omega=0.6 is NOT optimal on test F1. Gap to best test F1: %.4f (paper test_f1=%.4f, best test_f1=%.4f at omega=%.1f)",
            delta,
            paper_row["test_f1"],
            best_test_f1,
            best_test_omega,
        )
    else:
        logger.info("omega=0.6 is optimal on test F1.")


def main() -> int:
    t0 = time.monotonic()
    logger = setup_logging()
    input_paths: List[str] = []
    loaded_from_file = False
    recomputed = False

    try:
        labels = load_ground_truth()
        logger.info("Loaded ground truth labels from labels/CE.csv and labels/BE.csv")

        merged, searched, loaded_from_file = try_load_existing_scores(labels, logger)
        input_paths.extend(searched)

        if merged is not None and len(merged) >= 50:
            eval_df = merged
            logger.info("Using existing score file (>=50 labeled accounts with scores).")
        elif SCORE_CACHE.exists():
            logger.info("Loading cached scores from %s", SCORE_CACHE)
            cached = pd.read_csv(SCORE_CACHE)
            cached["address"] = cached["address"].map(normalize_address)
            eval_df = labels.merge(cached, on="address", how="inner")
            input_paths.append(str(SCORE_CACHE.relative_to(PROJECT_ROOT)))
            if eval_df.empty:
                raise RuntimeError("Cached score file has no overlap with CE/BE labels.")
        else:
            txs_file = DEFAULT_TXS_FILE
            input_paths.append(str(txs_file.relative_to(PROJECT_ROOT)))
            logger.info(
                "No suitable existing score file found; computing scores with SuspiciousAccountDetector."
            )
            all_scores = compute_scores_with_detector(txs_file, logger)
            all_scores.to_csv(SCORE_CACHE, index=False)
            logger.info("Cached computed scores to %s", SCORE_CACHE)
            recomputed = True
            eval_df = labels.merge(all_scores, on="address", how="inner")

        if eval_df.empty:
            raise RuntimeError(
                "No labeled accounts with both ground-truth labels and rule_score/model_score."
            )

        train, val, test = stratified_split(eval_df)
        results_df = run_omega_grid(train, val, test)
        results_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")

        save_plot(results_df)
        write_summary_log(
            logger,
            input_paths=input_paths,
            loaded_from_file=loaded_from_file,
            recomputed=recomputed,
            labels=eval_df,
            train=train,
            val=val,
            test=test,
            results_df=results_df,
            elapsed=time.monotonic() - t0,
        )
        logger.info("Results saved to %s", RESULTS_CSV)
        logger.info("Plot saved to %s", PLOT_PATH)
        return 0

    except Exception as exc:
        logger.error("Experiment failed: %s", exc, exc_info=True)
        logger.error(
            "FAILURE REASON: Unable to obtain rule_score and model_score for labeled accounts. %s",
            exc,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
