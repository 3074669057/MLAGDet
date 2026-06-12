"""Frozen paper protocol for clean main experiments."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results_main_clean_no_label_prior"
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = OUTPUT_ROOT / "logs"

THRESHOLD = 0.25
MODEL_WEIGHT = 0.6  # omega in paper: S = omega*S_ML + (1-omega)*S_rule
RULE_WEIGHT = 0.4
MIN_COMMUNITY_SIZE = 3
DISABLE_LABEL_PRIOR = True

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
OMEGAS = [round(x, 1) for x in [i / 10 for i in range(11)]]

PAPER_TABLE2_MLAGDET = {
    "precision": 0.9150,
    "recall": 0.9925,
    "f1": 0.9522,
    "auc": 0.9694,
    "ac": 0.3560,
    "nmi": 0.9174,
    "ari": 0.8734,
}
