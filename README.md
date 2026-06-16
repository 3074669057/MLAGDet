# MLAGDet

**MLAGDet: An Account-Community-Network Hierarchy for Candidate Laundering Organization Analysis Beyond Observed Transactions**

MLAGDet is an **Ethereum-oriented** account–community–network three-level AML analysis framework. It identifies suspicious accounts from observed on-chain transactions, reconstructs candidate laundering communities, and expands high-risk candidate members in a broader transaction neighborhood via APPR. This repository provides a **research reproduction pipeline** for the paper; CLI demos and optional online tracing are implementation-layer extensions documented later in this file.

> **Scope.** Empirical conclusions in the paper are validated on **Ethereum**, not on all blockchains. Do not treat this repository as a generic multi-chain AML product.

---

## Table of Contents

1. [Project Title and Paper Positioning](#1-project-title-and-paper-positioning)
2. [Method Overview](#2-method-overview)
3. [Paper Headline Results](#3-paper-headline-results)
4. [Citation](#4-citation)
5. [Installation and Environment](#5-installation-and-environment)
6. [Quick Start](#6-quick-start)
7. [Data Preparation](#7-data-preparation)
8. [Paper Reproduction Input Format](#8-paper-reproduction-input-format)
9. [Token Preprocessing](#9-token-preprocessing)
10. [Configuration](#10-configuration)
11. [Model Zoo and Checkpoints](#11-model-zoo-and-checkpoints)
12. [Reproducing Paper Tables and Figures](#12-reproducing-paper-tables-and-figures)
13. [FAQ and Troubleshooting](#13-faq-and-troubleshooting)
14. [Repository Structure](#14-repository-structure)
15. [Contribution](#15-contribution)
16. [License](#16-license)
17. [Implementation-Level Interfaces (CLI, API, Demo)](#17-implementation-level-interfaces-cli-api-demo)

---

## 1. Project Title and Paper Positioning

| Item | Description |
|------|-------------|
| **Paper title** | MLAGDet: An Account-Community-Network Hierarchy for Candidate Laundering Organization Analysis Beyond Observed Transactions |
| **Task domain** | Ethereum AML analysis |
| **Account level** | Suspicious-account detection (rule + ML fusion) |
| **Community level** | Candidate laundering-group discovery |
| **Network level** | APPR-based candidate expansion beyond observed seeds |
| **Empirical scope** | Conclusions are validated on **Ethereum** transaction data, not all blockchains |

The repository implements the paper hierarchy as an offline-first reproduction pipeline. Optional CLI, Python API, synthetic demo, and online Scrapy tracing are **engineering conveniences** — not substitutes for full paper experiments.

---

## 2. Method Overview

MLAGDet follows an **account → community → network** hierarchy:

| Level | Paper role | Implementation alias (code) |
|-------|------------|------------------------------|
| **Account** | Feature extraction, 14 risk rules, RF / XGBoost / LightGBM ensemble, rule + ML fusion, suspicious scoring | `mlagdet.point` |
| **Community** | Louvain, LPA, pseudo-likelihood partitioning; Jaccard-based fusion / merge | `mlagdet.line` |
| **Network** | APPR expansion; candidate re-detection / re-verification | `mlagdet.surface` |

**Account-level method:** Random Forest, XGBoost, LightGBM, ensemble learning, 14 configurable risk rules, rule + ML fusion.

**Community-level method:** Louvain, LPA (Label Propagation), pseudo-likelihood optimization, Jaccard-based fusion / merge across algorithm outputs.

**Network-level method:** APPR (Approximate Personalized PageRank) expansion on a local transaction graph, followed by candidate re-detection / re-verification.

**Input graph mode (paper):** weighted, directed, temporal, token-aware account transaction graph.

---

## 3. Paper Headline Results

### Account-level (paper Table headline)

| Metric | Value |
|--------|-------|
| Precision | 91.50% |
| Recall | 99.25% |
| F1 | 95.22% |
| AUC | 0.9694 |

### Group-level (paper Table headline)

| Metric | Value |
|--------|-------|
| AC | 0.3560 |
| NMI | 0.9174 |
| ARI | 0.8734 |

> Group-level metrics in the paper are computed against **reference community partitions derived from weak supervision**, not full ground-truth laundering organizations. See [FAQ](#13-faq-and-troubleshooting).

### Scalability / runtime (paper)

| Stage | Detail |
|-------|--------|
| Stage-1 at 100% scale | 461,712 transactions, 74,732 nodes, 86,534 edges |
| Runtime | 354.05 s |
| Peak RSS | 1232 MB |
| APPR expansion | 9.40 s |
| Re-verification | 452.61 s / 168,436 rows |

---

## 4. Citation

If you use MLAGDet in research, please cite the paper (update with published venue when available):

```bibtex
@article{mlagdet2026,
  title   = {MLAGDet: An Account-Community-Network Hierarchy for Candidate Laundering Organization Analysis Beyond Observed Transactions},
  author  = {MLAGDet Authors},
  year    = {2026},
  note    = {Software available at https://github.com/3074669057/MLAGDet}
}
```

See also `CITATION.cff` for software metadata.

---

## 5. Installation and Environment

The **paper does not specify** exact Python, PyTorch, or CUDA versions.

**Recommended reproduction environment:**

| Component | Version / note |
|-----------|----------------|
| Python | 3.10 |
| pip | ≥ 23 |
| PyTorch | 2.5.0 (optional; not required for tree-based account models) |
| scikit-learn | via `requirements.txt` |
| xgboost | optional extra `.[ml]` |
| lightgbm | optional extra `.[ml]` |
| numpy / pandas / scipy | via `requirements.txt` |
| networkx | via `requirements.txt` |

```bash
cd MLAGDet
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

### Optional extras (defined in `pyproject.toml`)

```bash
pip install -e ".[online]"   # Scrapy + web3 for optional online tracing
pip install -e ".[ml]"       # LightGBM + XGBoost for model training
pip install -e ".[dev]"      # pytest for testing
```

Core dependencies are listed in `requirements.txt` and mirrored in `[project.dependencies]` inside `pyproject.toml`.

---

## 6. Quick Start

**Paper reproduction** requires the full Ethereum dataset (see [Data Preparation](#7-data-preparation)). For a **smoke test** of the installable package only:

```bash
mlagdet demo --output outputs/demo
```

This runs the synthetic minimal example and writes outputs under `outputs/demo/`. It validates installation and APIs; it **does not** reproduce paper headline metrics.

Full-pipeline evaluation on labeled data:

```bash
python scripts/eval_pipeline.py --input transactions --labels-dir labels --output outputs/eval
```

---

## 7. Data Preparation

### Label sources

| Source | Role |
|--------|------|
| **EthereumHeist** | Malicious / benign address labels |
| **Real-CATS** | Additional labeled addresses and evaluation context |

### Crawling and corpus (paper)

| Item | Value / note |
|------|--------------|
| Transaction-subgraph crawling | TRacer-style tracing / related crawler logic |
| Full crawled corpus | 6,556,352 transaction records |
| Labeled addresses | 12,636 malicious, 16,021 benign |
| Account-level filtered evaluation subset | 461,712 transactions, 74,732 accounts |
| Labeled laundering addresses in subset | 2,941 |
| Unlabeled accounts in subset | 71,791 |
| Train / validation / test split | **6 : 2 : 2** |

Full Real-CATS-derived corpora are **not redistributed** in this repository. Download and layout instructions:

```bash
python scripts/download_data.py
```

Place transaction CSVs under `transactions/` and label files (`CE.csv`, `BE.csv`, etc.) under `labels/`. See `docs/reproducibility.md` for additional notes.

---

## 8. Paper Reproduction Input Format

Paper reproduction expects a **weighted, directed, temporal, token-aware** account transaction graph encoded as CSV rows.

### Paper reproduction input schema

| Field | Required for paper repro | Description |
|-------|--------------------------|-------------|
| `from` | Yes | Sender account |
| `to` | Yes | Receiver account |
| `value` | Yes | Raw transfer amount (token-native units) |
| `timestamp` | Yes | Transaction time (Unix epoch recommended) |
| `token_symbol` | Yes | Token symbol (e.g. ETH, USDT) |
| `contract_address` | Yes | ERC-20 contract address (empty or sentinel for native ETH) |
| `tx_hash` | Yes | Transaction hash |
| `decimals` | Yes | Token decimals for normalization |

Example minimal demo (subset of fields): `examples/minimal/transactions_sample.csv`

Detailed field notes: `docs/data_format.md`

---

## 9. Token Preprocessing

| Rule | Detail |
|------|--------|
| ETH | 18 decimals |
| Canonical ERC-20 (USDT, USDC, DAI, WETH, WBTC, UNI, LINK, …) | Use standard decimals **only when mainnet contract addresses strictly match** canonical deployments |
| Non-canonical lookalike tokens | Must **not** be treated as canonical stablecoins |
| Implementation | Token-aware and decimal-normalized |
| Price normalization | **Not** fully price-aware unless code explicitly implements price normalization |

When reproducing paper numbers, verify token metadata columns are present and canonical contract matching is enforced in your preprocessing stage.

---

## 10. Configuration

Default YAML: `configs/default.yaml`  
Rule definitions: `configs/rules_config.json`  
Blacklist: `configs/blacklist.json`

### Paper default parameters

| Parameter | Symbol / name | Paper default |
|-----------|---------------|---------------|
| Initial suspicious threshold | δ | 0.25 |
| ML fusion weight | ω | 0.6 |
| Rule fusion weight | 1 − ω | 0.4 |
| APPR alpha | α | 0.15 |
| APPR epsilon | ε | 1e-3 |
| APPR beta | β | 0.8 |
| Re-verification threshold | δ_re | 0.37 |
| Re-verification fusion | rule / ML | 0.5 / 0.5 |
| Jaccard merge threshold | η | 0.5 |
| Minimum community size | s_min | 3 |

> Implementation config keys may use names such as `rule_weight` / `model_weight`. Align values with paper ω and 1−ω when reproducing headline results. Some APPR-related defaults in `configs/default.yaml` may differ from paper α/β — override via CLI flags or experiment scripts where supported.

### What the paper does **not** specify

| Topic | Status |
|-------|--------|
| Training epochs | **Not specified** |
| Batch size | **Not specified** |
| Optimizer | **Not specified** |
| Learning-rate schedule | **Not specified** |
| RF / XGBoost / LightGBM internal hyperparameters | **Implementation-specific** unless defined in config files |
| Official checkpoint download links | **Not provided** by the paper |

---

## 11. Model Zoo and Checkpoints

| Model / artifact | Status |
|------------------|--------|
| Random Forest | Train locally (`scripts/train_models.py`) or use rule-only fallback |
| XGBoost | Train locally; requires `.[ml]` |
| LightGBM | Train locally; requires `.[ml]` |
| Official pretrained checkpoints | **Not provided** unless a future release adds them |
| Default `models/` directory | Empty in source; detection falls back to rule engine weights |

**If no release weights exist, train from scratch** on Real-CATS / EthereumHeist-derived features before expecting paper-level account metrics.

---

## 12. Reproducing Paper Tables and Figures

### Paper table / figure reproduction mapping

| Paper content | Script | Main output | Status |
|---------------|--------|-------------|--------|
| Account-level main results | `scripts/eval_pipeline.py` | `account_metrics.json` | Available |
| Group-level main results | `scripts/eval_pipeline.py` | `group_metrics.json` | Partial — inter-algorithm NMI/ARI only unless full eval aligned |
| Threshold sensitivity | `scripts/reproduce/run_threshold_experiment.py` | `threshold_sweep.csv` / plots | Available |
| APPR alpha sensitivity | `scripts/reproduce/run_alpha_sensitivity_experiment.py` | sensitivity CSV / PNG | Available |
| APPR beta sensitivity | — | sensitivity CSV / PNG | **Planned / implementation to be aligned** |
| Seed degradation | `scripts/reproduce/run_seed_degradation_experiment.py` | degradation CSV / PNG | Available |
| Offline APPR robustness | `scripts/reproduce/run_offline_appr_seed_degradation.py` | APPR robustness outputs | Available |
| Ablation | — | ablation CSV | **Planned / TODO** |
| Case studies | — | case study summaries | **Planned / TODO** |

See `docs/reproducibility.md` for experiment-script details.

Example workflow:

```bash
# Headline account + community pipeline metrics (requires full data)
python scripts/eval_pipeline.py --input transactions --labels-dir labels --output outputs/paper_eval

# Threshold sweep
python scripts/reproduce/run_threshold_experiment.py

# APPR alpha sensitivity
python scripts/reproduce/run_alpha_sensitivity_experiment.py
```

---

## 13. FAQ and Troubleshooting

**1. Why does the demo run with only `from` / `to` / `value` / `timestamp`, but paper reproduction needs more fields?**  
The bundled demo (`mlagdet demo`) uses a **synthetic smoke-test CSV** to validate installation. Paper reproduction requires the full token-aware schema (§8) on Ethereum-scale data.

**2. Does the paper specify epochs, batch size, optimizer, or learning-rate schedule?**  
No. The paper does not specify epochs, batch size, optimizer, or learning-rate schedule. Tree-based account models use implementation defaults unless you set them in config.

**3. Are official pretrained checkpoints provided?**  
No. The paper does not provide official checkpoint download links. Train models locally or run rule-only mode.

**4. Why are group-level metrics not full organization-level ground truth?**  
Paper group metrics (AC, NMI, ARI) compare detected communities to **reference partitions from weak supervision** (algorithm consensus / label-derived structure), not to complete laundering-organization ground truth.

**5. What should I check if reproduced metrics differ from the paper?**  
Verify: (a) full 461k-transaction filtered subset, (b) 6:2:2 split, (c) token decimals and canonical contracts, (d) paper default parameters (δ, ω, APPR α/β/ε, δ_re, η, s_min), (e) trained ensemble models present, (f) label file layout under `labels/`.

**6. What should I check if online tracing / Etherscan crawling fails?**  
Install `.[online]`, copy `.env.example` → `.env`, set `ETH_RPC_URL` and `ETHERSCAN_API_KEY`, install BlockchainSpider separately, and see `docs/online_trace.md`. Online tracing is optional and not required for offline paper reproduction.

**7. How should token decimals and canonical tokens be handled?**  
ETH uses 18 decimals. Map canonical ERC-20 tokens to standard decimals only when `contract_address` matches mainnet canonical deployments. Reject lookalike tokens with similar symbols but different contracts.

---

## 14. Repository Structure

| Path | Purpose |
|------|---------|
| `src/mlagdet/` | Installable package |
| `src/mlagdet/point/` | **Account-level** implementation alias |
| `src/mlagdet/line/` | **Community-level** implementation alias |
| `src/mlagdet/surface/` | **Network-level** implementation alias |
| `configs/` | Default YAML, rules, blacklist |
| `scripts/` | Training, eval, data download |
| `scripts/reproduce/` | Paper sensitivity / robustness experiments |
| `scripts/eval_pipeline.py` | Headline metrics JSON export wrapper |
| `examples/minimal/` | Synthetic offline demo |
| `docs/` | Data format, reproducibility, online trace |
| `tests/` | pytest suite |

```text
MLAGDet/
├── src/mlagdet/
│   ├── point/            # account-level alias
│   ├── line/             # community-level alias
│   ├── surface/          # network-level alias
│   ├── cli.py
│   └── ...
├── configs/
├── scripts/
│   ├── eval_pipeline.py
│   └── reproduce/
├── examples/minimal/
├── docs/
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 15. Contribution

Contributions are welcome. See `CONTRIBUTING.md`. Typical workflow:

1. Fork the repository  
2. Create a feature branch  
3. Add tests for behavioral changes  
4. Run `pip install -e ".[dev]"` and `pytest`  
5. Open a pull request  

---

## 16. License

**MIT License — see [LICENSE](LICENSE).**  
License terms are defined by the repository `LICENSE` file, not by the paper.

---

## 17. Implementation-Level Interfaces (CLI, API, Demo)

The sections below describe **engineering interfaces**. They are not the paper's core contribution.

### Unified CLI

```bash
python -m mlagdet --help
mlagdet --help

# Offline smoke-test demo (synthetic data)
mlagdet demo --output outputs/demo

# Detection on custom input
mlagdet detect \
  --input examples/minimal/transactions_sample.csv \
  --config configs/default.yaml \
  --output outputs/detect

# Offline APPR trace from a community JSON
mlagdet trace \
  --community outputs/demo/point_line/combined/transactions_sample_community_0.json \
  --transactions examples/minimal/transactions_sample.csv \
  --output outputs/trace
```

### Online tracing (optional extension)

Requires RPC/API configuration and BlockchainSpider:

```bash
pip install -e ".[online]"
mlagdet trace --community community.json --online --output outputs/trace_online
```

See `docs/online_trace.md`.

### Python API

```python
from mlagdet import MoneyLaunderingDetector, expand_community_offline

detector = MoneyLaunderingDetector(
    in_dir="examples/minimal",
    out_dir="outputs/api_detect",
)
detector.run()

appr = expand_community_offline(
    "examples/minimal/transactions_sample.csv",
    seed_addresses=["0xaaa1111111111111111111111111111111111111"],
    top_k=20,
)
```

Module-level imports (code aliases):

```python
from mlagdet.point import SuspiciousAccountDetector      # account
from mlagdet.line import CommunityDetector               # community
from mlagdet.surface import expand_community_offline     # network
```

### Minimal offline example

```bash
python examples/minimal/run_demo.py
```

Files: `examples/minimal/transactions_sample.csv`, `examples/minimal/labels/`, `examples/minimal/run_demo.py`

### Train custom models (optional)

```bash
pip install -e ".[ml]"
python scripts/train_models.py --help
```

### Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Synthetic demo data

The demo dataset is **small and synthetic**. Metrics from `mlagdet demo` must not be compared to paper benchmarks.
