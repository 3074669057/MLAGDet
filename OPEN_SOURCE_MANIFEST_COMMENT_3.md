# Open-Source Manifest — Reviewer Comment 3 Package

> Auto-generated companion to `dist/MLAGDet-review-comment-3-MANIFEST.txt`.  
> Run `python scripts/package_comment3_release.py` to refresh the zip and manifest.

---

## 1. Packaged Files (intended)

### Root pipeline

| Path | Role |
|------|------|
| `money_laundering_detector.py` | Stage 1: account-level detection + community discovery |
| `run_analysis.py` | Stage 2: APPR expansion orchestration |
| `suspicious_account_detector.py` | Stage 3: detection on expanded graph |
| `run_threshold_experiment.py` | Threshold sweep helper |
| `rules_config.json` | 14 AML rule definitions |
| `README_rules_config.md` | Rule configuration reference |
| `requirements.txt` | Python dependencies |
| `blacklist.json` | Known suspicious address list (runtime) |
| `whitelist.json` | Known benign address list (runtime) |
| `.env.example` | Placeholder RPC/API env vars |
| `REPRODUCIBILITY_COMMENT_3.md` | Reproducibility guide (this release) |
| `OPEN_SOURCE_MANIFEST_COMMENT_3.md` | File manifest and exclusions |
| `TODO_FIGURE_TERMINOLOGY_COMMENT_3.md` | Figures still using legacy terminology |

### Sample data

| Path | Role |
|------|------|
| `examples/comment3_sample/sample_transactions.csv` | Minimal transaction CSV (10 rows) |
| `examples/comment3_sample/sample_CE.csv` | Sample criminal-entity labels |
| `examples/comment3_sample/sample_BE.csv` | Sample benign-entity labels |
| `examples/comment3_sample/sample_community.json` | APPR seed task JSON |
| `examples/comment3_sample/sample_importance/*.csv` | Sample APPR importance files |

### Experiments

| Path | Role |
|------|------|
| `experiment/main_clean_no_label_prior/*.py` | Frozen clean main protocol |
| `experiment/phase1_seed_degradation_experiment.py` | Online seed degradation |
| `experiment/phase1_offline_appr_seed_degradation.py` | Offline APPR seed degradation |
| `experiment/phase1_appr_coverage_runner.py` | APPR coverage scanner |
| `experiment/phase2/phase2_alpha_sensitivity_experiment.py` | Fusion-weight sensitivity |
| `experiment/_regenerate_scalability_summary.py` | Scalability summary helper |
| `experiment/reviewer2_major1_ablation/ablation_utils.py` | Shared metrics helpers |

### Spider (APPR/TTR)

| Path | Role |
|------|------|
| `Spider/scrapy.cfg` | Scrapy project config |
| `Spider/requirements.txt` | Spider dependencies |
| `Spider/BlockchainSpider/**` | Full crawler package (env-based settings) |

### Real-CATS (training only)

| Path | Role |
|------|------|
| `Real-CATS-master/train_money_laundering_models.py` | Train ML models |
| `Real-CATS-master/requirements.txt` | Training dependencies |
| `Real-CATS-master/README.md` | Dataset description |
| `Real-CATS-master/models/token_q95_map.json` | Token normalization map |

---

## 2. Explicitly Excluded

| Path / pattern | Reason |
|----------------|--------|
| `transactions/labeled_transactions.csv` (~150 MB) | Full paper dataset |
| `labels/CE.csv`, `labels/BE.csv` (full) | Full label sets |
| `Real-CATS-master/*.tsv` | Large training tables (7–50 MB each) |
| `Real-CATS-master/models/*.pkl` | Large model files; retrain or obtain separately |
| `results*/`, `experiment/phase1/`, `experiment/phase2/alpha_sensitivity/` outputs | Generated artifacts |
| `Spider/0x*/`, `Spider/.scrapy/`, `Spider/*/output/` | Crawl outputs and HTTP cache |
| `experiment/phase_scalability/` | Scalability inputs/outputs (GB-scale) |
| `experiment/token_wise_normalization_robustness_analysis/` | Unrelated reviewer concern / 4+ GB |
| `__pycache__/`, `*.log`, `*.pkl`, `*.zip`, `.env` | Cache, secrets, archives |
| Files > 10 MB | Size cap for lightweight release |

---

## 3. Zip Size

See `dist/MLAGDet-review-comment-3-MANIFEST.txt` after running the packaging script for:

- Total zip bytes / MB
- Per-file listing
- Files skipped due to size
- Pre-zip validation results

**Target:** < 20 MB (hard limit check at 50 MB).

---

## 4. Files Over 10 MB

None expected in this curated package. Any file exceeding 10 MB during packaging is skipped and logged in the manifest.

---

## 5. Security Review

| Check | Status |
|-------|--------|
| Hardcoded API keys removed from `Spider/BlockchainSpider/settings.py` | Yes — env vars only |
| RPC endpoints in source | Removed from packaged settings |
| `.env` included | No — only `.env.example` |
| Private paths in scripts | Sample paths use relative project paths |
| Real Etherscan/Alchemy credentials | **Were present in original settings.py; sanitized before release** |

---

## 6. Legacy Terminology in Text Files

Packaged markdown uses **account--community--network hierarchy** terminology.

Legacy terms may still appear in:

- Root `README.md` (not bundled by default; see TODO figure list)
- `1.txt` experiment report (if bundled)
- Script docstrings in Phase-1/Phase-2 (e.g., "point-level")

See `TODO_FIGURE_TERMINOLOGY_COMMENT_3.md` for figures (`image/fig2.png`, `image/fig3.png`) not included in this zip.

---

## 7. Items Needing Author Confirmation

1. **Figure terminology** — manual update required for `image/fig2.png`, `image/fig3.png` (see `TODO_FIGURE_TERMINOLOGY_COMMENT_3.md`).
2. **ML model files** — not bundled; confirm whether authors will publish separately.
3. **Legacy APPR task JSON** — `Spider/file.json` still uses β=0.6, ε=0.0001; treat as legacy, not final pipeline default.

---

## 8. Regenerate Package

```bash
python scripts/package_comment3_release.py
```

Outputs:

- `dist/MLAGDet-review-comment-3-package/` (staging directory)
- `dist/MLAGDet-review-comment-3-package.zip`
- `dist/MLAGDet-review-comment-3-MANIFEST.txt`
