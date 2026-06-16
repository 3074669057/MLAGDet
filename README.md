# MLAGDet

**MLAGDet: An Account-Community-Network Hierarchy for Candidate Laundering Organization Analysis Beyond Observed Transactions**

MLAGDet is an **Ethereum-oriented anti-money-laundering (AML) research framework** for detecting suspicious accounts, discovering candidate laundering communities, and expanding high-risk members beyond observed transactions.

This repository provides the official implementation and reproduction pipeline for the MLAGDet paper. It is designed for **academic reproducibility**, not as a production compliance system or a generic multi-chain AML product.

---

## Overview

MLAGDet follows a three-level analysis hierarchy:

| Level | Goal | Main methods |
|---|---|---|
| Account | Detect suspicious accounts | Risk rules, RF, XGBoost, LightGBM, rule-ML fusion |
| Community | Discover candidate laundering groups | Louvain, LPA, pseudo-likelihood partitioning, Jaccard fusion |
| Network | Expand high-risk candidates | APPR expansion and re-verification |

The paper experiments are validated on **Ethereum transaction data**.

---

## Main Results

### Account-level detection

| Precision | Recall | F1 | AUC |
|---:|---:|---:|---:|
| 91.50% | 99.25% | 95.22% | 0.9694 |

### Community-level discovery

| AC | NMI | ARI |
|---:|---:|---:|
| 0.3560 | 0.9174 | 0.8734 |

### Scalability

| Stage | Scale / runtime |
|---|---|
| Stage-1 evaluation | 461,712 transactions, 74,732 nodes, 86,534 edges |
| Runtime | 354.05 s |
| Peak RSS | 1232 MB |
| APPR expansion | 9.40 s |
| Re-verification | 452.61 s / 168,436 rows |

---

## Installation

```bash
git clone https://github.com/3074669057/MLAGDet.git
cd MLAGDet

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate     # Windows

python -m pip install --upgrade pip
pip install -e .
```

Optional extras:

```bash
pip install -e ".[ml]"       # XGBoost / LightGBM training
pip install -e ".[online]"   # Optional online tracing
pip install -e ".[dev]"      # Tests
```

Recommended environment: Python 3.10. Exact CUDA / PyTorch settings are not required for the tree-based account models.

---

## Quick Start

Run a minimal smoke test:

```bash
mlagdet demo --output outputs/demo
```

The demo uses synthetic data and only verifies installation and APIs. It does **not** reproduce the paper results.

Run the offline evaluation pipeline:

```bash
python scripts/eval_pipeline.py \
  --input transactions \
  --labels-dir labels \
  --output outputs/paper_eval
```

---

## Data Preparation

The full Real-CATS-derived corpus is **not redistributed** in this repository.

Expected data layout:

```text
MLAGDet/
├── transactions/
│   └── *.csv
└── labels/
    ├── CE.csv
    ├── BE.csv
    └── ...
```

Paper reproduction expects a weighted, directed, temporal, token-aware Ethereum transaction graph.

Required transaction fields:

| Field | Description |
|---|---|
| `from` | Sender address |
| `to` | Receiver address |
| `value` | Raw transfer amount |
| `timestamp` | Transaction time |
| `token_symbol` | Token symbol |
| `contract_address` | Token contract address |
| `tx_hash` | Transaction hash |
| `decimals` | Token decimals |

See `docs/data_format.md` and `docs/reproducibility.md` for details.

---

## Configuration

Main configuration files:

```text
configs/default.yaml
configs/rules_config.json
configs/blacklist.json
```

Important paper parameters:

| Parameter | Value |
|---|---:|
| Initial suspicious threshold `δ` | 0.25 |
| ML fusion weight `ω` | 0.6 |
| Rule fusion weight `1 - ω` | 0.4 |
| APPR alpha `α` | 0.15 |
| APPR epsilon `ε` | 1e-3 |
| APPR beta `β` | 0.8 |
| Re-verification threshold `δ_re` | 0.37 |
| Jaccard merge threshold `η` | 0.5 |
| Minimum community size `s_min` | 3 |

Some implementation defaults may differ from paper settings. Override them explicitly when reproducing reported results.

---

## Reproducing Experiments

| Experiment | Script |
|---|---|
| Main account / community metrics | `scripts/eval_pipeline.py` |
| Threshold sensitivity | `scripts/reproduce/run_threshold_experiment.py` |
| APPR alpha sensitivity | `scripts/reproduce/run_alpha_sensitivity_experiment.py` |
| Seed degradation | `scripts/reproduce/run_seed_degradation_experiment.py` |
| Offline APPR robustness | `scripts/reproduce/run_offline_appr_seed_degradation.py` |

Example:

```bash
python scripts/eval_pipeline.py \
  --input transactions \
  --labels-dir labels \
  --output outputs/paper_eval
```

---

## Repository Structure

```text
MLAGDet/
├── src/mlagdet/
│   ├── point/        # Account-level detection
│   ├── line/         # Community-level discovery
│   └── surface/      # Network-level expansion
├── configs/          # Default parameters and rules
├── scripts/          # Training, evaluation, reproduction scripts
├── examples/         # Minimal demo data
├── docs/             # Data format and reproducibility notes
└── tests/            # Unit tests
```

---

## Citation

If you use this repository, please cite:

```bibtex
@article{mlagdet2026,
  title   = {MLAGDet: An Account-Community-Network Hierarchy for Candidate Laundering Organization Analysis Beyond Observed Transactions},
  author  = {MLAGDet Authors},
  year    = {2026},
  note    = {Software available at https://github.com/3074669057/MLAGDet}
}
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Disclaimer

MLAGDet is a research prototype for academic study. It should not be used as the sole basis for legal, financial, or compliance decisions.
