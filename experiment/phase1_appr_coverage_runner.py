#!/usr/bin/env python3
"""
Phase-1 APPR coverage runner: identify communities that need more APPR importance
files and optionally invoke run_analysis.py to expand Spider coverage for the
seed degradation experiment.

Supports reduced community JSON files to avoid running APPR on full large
communities.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Allow importing sibling experiment modules when invoked as a script.
_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from phase1_seed_degradation_experiment import (  # noqa: E402
    _extract_seed_from_path,
    discover_community_files,
    load_label_addresses,
    normalize_address,
    parse_community_accounts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLAN_CSV = "phase1_appr_coverage_plan.csv"
RUN_RESULTS_CSV = "phase1_appr_coverage_run_results.csv"
LOG_TXT = "phase1_appr_coverage_log.txt"


# ---------------------------------------------------------------------------
# APPR seed discovery
# ---------------------------------------------------------------------------


def list_importance_paths(spider_dir: Path) -> List[Path]:
    """Recursively list all importance/*.csv files under Spider."""
    if not spider_dir.is_dir():
        return []
    return sorted(spider_dir.rglob("importance/*.csv"))


def list_appr_seed_addresses(spider_dir: Path) -> Set[str]:
    """Return seed addresses that already have APPR importance CSV files."""
    seeds: Set[str] = set()
    for path in list_importance_paths(spider_dir):
        seed = _extract_seed_from_path(path)
        if seed:
            seeds.add(seed)
    return seeds


# ---------------------------------------------------------------------------
# Community inventory
# ---------------------------------------------------------------------------


def build_community_inventory(
    community_files: List[Path],
    ce_labels: Set[str],
    be_labels: Set[str],
    appr_seeds: Set[str],
) -> pd.DataFrame:
    """Scan all community JSON files and collect coverage statistics."""
    rows: List[Dict[str, Any]] = []

    for json_path in community_files:
        community_id = json_path.stem
        try:
            accounts = parse_community_accounts(json_path)
        except Exception as exc:
            logging.warning("Skip %s: parse error - %s", community_id, exc)
            continue

        malicious = accounts & ce_labels
        benign = accounts & be_labels
        existing_appr = sum(1 for addr in malicious if addr in appr_seeds)

        rows.append(
            {
                "community_id": community_id,
                "community_json_path": str(json_path.resolve()),
                "community_size": len(accounts),
                "malicious_seed_count": len(malicious),
                "benign_seed_count": len(benign),
                "existing_appr_malicious_count": existing_appr,
            }
        )

    return pd.DataFrame(rows)


def select_communities_for_run(
    inventory: pd.DataFrame,
    min_malicious_seeds: int,
    max_community_size: int,
    target_appr_per_community: int,
    max_communities: int,
) -> pd.DataFrame:
    """
    Mark communities that need more APPR coverage and pick top candidates.
    Priority: higher malicious_seed_count, then smaller community_size.
    """
    if inventory.empty:
        inventory = inventory.copy()
        inventory["selected"] = False
        inventory["selection_rank"] = pd.NA
        return inventory

    plan = inventory.copy()
    eligible = (
        (plan["malicious_seed_count"] >= min_malicious_seeds)
        & (plan["community_size"] <= max_community_size)
        & (plan["existing_appr_malicious_count"] < target_appr_per_community)
    )
    plan["selected"] = False
    plan["selection_rank"] = pd.NA

    candidates = plan[eligible].sort_values(
        by=["malicious_seed_count", "community_size"],
        ascending=[False, True],
    )
    selected_idx = candidates.head(max_communities).index
    plan.loc[selected_idx, "selected"] = True
    for rank, idx in enumerate(selected_idx, start=1):
        plan.at[idx, "selection_rank"] = rank

    return plan


# ---------------------------------------------------------------------------
# Reduced community JSON
# ---------------------------------------------------------------------------


def load_community_json_raw(json_path: Path) -> Any:
    """Load raw community JSON without transformation."""
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def select_ce_seeds_for_community(
    community_json_path: Path,
    ce_labels: Set[str],
    appr_seeds: Set[str],
    max_seeds: int,
) -> List[str]:
    """
    Pick CE malicious seeds for APPR, preferring addresses without importance CSV.
    """
    accounts = parse_community_accounts(community_json_path)
    malicious = sorted(accounts & ce_labels)
    if not malicious:
        return []

    missing_appr = [addr for addr in malicious if addr not in appr_seeds]
    with_appr = [addr for addr in malicious if addr in appr_seeds]
    ordered = missing_appr + with_appr
    return ordered[:max_seeds]


def _item_source_address(item: Any) -> Optional[str]:
    """Extract source address from one JSON list item."""
    if not isinstance(item, dict):
        return normalize_address(item)
    for key in ("source", "address", "node", "id", "account", "from", "out"):
        if key in item:
            addr = normalize_address(item[key])
            if addr:
                return addr
    return None


def _make_source_out_entry(seed: str, template: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build one run_analysis-compatible list entry for a seed address."""
    if template is not None:
        entry = dict(template)
        entry["source"] = seed
        entry["out"] = seed
        return entry
    return {
        "source": seed,
        "out": seed,
        "types": "external,internal,erc20,erc721",
    }


def build_reduced_community_json(
    original_data: Any,
    selected_seeds: List[str],
) -> List[Dict[str, Any]]:
    """
    Build a reduced JSON list compatible with run_analysis.py / Scrapy TTR spider.

    run_analysis expects a list of dicts; each dict must provide at least:
      - source: APPR seed address
      - out: output directory name (typically the seed address)
      - types: comma-separated transaction types
    """
    seed_set = set(selected_seeds)
    if not seed_set:
        raise ValueError("selected_seeds is empty")

    if isinstance(original_data, list):
        reduced: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        template_by_seed: Dict[str, Dict[str, Any]] = {}

        for item in original_data:
            addr = _item_source_address(item)
            if addr and addr in seed_set and isinstance(item, dict):
                template_by_seed[addr] = item

        for seed in selected_seeds:
            if seed in template_by_seed:
                entry = _make_source_out_entry(seed, template_by_seed[seed])
            else:
                entry = _make_source_out_entry(seed, None)
            reduced.append(entry)
            seen.add(seed)

        return reduced

    if isinstance(original_data, dict):
        # Fallback: synthesize source/out entries when structure is not a flat list.
        template = None
        for key in ("nodes", "accounts", "members", "addresses", "community"):
            val = original_data.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                template = val[0]
                break
        return [_make_source_out_entry(seed, template) for seed in selected_seeds]

    raise ValueError(f"Unsupported community JSON top-level type: {type(original_data)}")


def reduced_json_filename(community_id: str, seed_count: int) -> str:
    """Filename for a reduced community JSON."""
    suffix = community_id
    prefix = "labeled_transactions_community_"
    if community_id.startswith(prefix):
        suffix = community_id[len(prefix) :]
    return f"reduced_labeled_transactions_community_{suffix}_seeds_{seed_count}.json"


def write_reduced_json(
    reduced_data: List[Dict[str, Any]],
    temp_dir: Path,
    community_id: str,
) -> Path:
    """Write reduced JSON to temp_dir without overwriting unrelated files."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_path = temp_dir / reduced_json_filename(community_id, len(reduced_data))
    if out_path.exists():
        logging.info("Reduced JSON already exists, reuse: %s", out_path)
        return out_path
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(reduced_data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    logging.info("Wrote reduced JSON (%d seeds): %s", len(reduced_data), out_path)
    return out_path


def stage_json_for_spider(json_path: Path, spider_dir: Path) -> Path:
    """
    Copy JSON into Spider/ so run_analysis.py can pass basename to Scrapy.

    run_analysis.py chdirs to Spider/ and invokes scrapy with file=<basename> only.
    """
    spider_dir.mkdir(parents=True, exist_ok=True)
    staged = spider_dir / json_path.name
    if staged.resolve() != json_path.resolve():
        shutil.copy2(json_path, staged)
        logging.info("Staged JSON for Scrapy: %s", staged)
    return staged


def prepare_execution_json(
    community_id: str,
    original_json_path: Path,
    ce_labels: Set[str],
    appr_seeds: Set[str],
    use_reduced_json: bool,
    max_seeds_per_community: int,
    temp_dir: Path,
    spider_dir: Path,
) -> Tuple[Path, bool, List[str]]:
    """
    Resolve the JSON path to feed into run_analysis.py.
    Returns (execution_json_path, used_reduced_json, selected_seeds).
    """
    if not use_reduced_json:
        return original_json_path.resolve(), False, []

    selected_seeds = select_ce_seeds_for_community(
        community_json_path=original_json_path,
        ce_labels=ce_labels,
        appr_seeds=appr_seeds,
        max_seeds=max_seeds_per_community,
    )
    if not selected_seeds:
        raise ValueError(f"No CE seeds available for {community_id}")

    original_data = load_community_json_raw(original_json_path)
    reduced_data = build_reduced_community_json(original_data, selected_seeds)
    reduced_path = write_reduced_json(reduced_data, temp_dir, community_id)
    stage_json_for_spider(reduced_path, spider_dir)
    return reduced_path.resolve(), True, selected_seeds


# ---------------------------------------------------------------------------
# run_analysis execution
# ---------------------------------------------------------------------------


def run_analysis_for_community(
    community_json_path: Path,
    timeout_seconds: int,
) -> Tuple[str, int, str]:
    """
    Invoke run_analysis.py for one community JSON.
    Returns (status, exit_code, error_message).
    """
    run_analysis_script = PROJECT_ROOT / "run_analysis.py"
    if not run_analysis_script.is_file():
        return "error", -1, f"run_analysis.py not found: {run_analysis_script}"

    if not community_json_path.is_file():
        return "error", -1, f"Community JSON not found: {community_json_path}"

    cmd = [
        sys.executable,
        str(run_analysis_script),
        "-a",
        str(community_json_path.resolve()),
    ]
    logging.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if result.returncode == 0:
            return "success", result.returncode, ""
        tail = (result.stderr or result.stdout or "").strip()[-500:]
        return "error", result.returncode, tail or f"exit code {result.returncode}"

    except subprocess.TimeoutExpired as exc:
        msg = f"timeout after {timeout_seconds}s"
        if exc.stderr:
            msg += f"; stderr tail: {str(exc.stderr)[-200:]}"
        return "timeout", -1, msg
    except Exception as exc:
        return "error", -1, str(exc)


def execute_coverage_plan(
    plan: pd.DataFrame,
    spider_dir: Path,
    ce_labels: Set[str],
    appr_seeds: Set[str],
    use_reduced_json: bool,
    max_seeds_per_community: int,
    temp_dir: Path,
    sleep_seconds: int,
    timeout_minutes: int,
    execute_one_first: bool,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Run run_analysis.py for selected communities and record outcomes."""
    selected = plan[plan["selected"] == True].sort_values("selection_rank")  # noqa: E712
    if execute_one_first:
        selected = selected.head(1)
        logging.info("execute_one_first: running only selection_rank=1")

    timeout_seconds = int(timeout_minutes * 60)
    run_rows: List[Dict[str, Any]] = []
    seed_plan_logs: List[Dict[str, Any]] = []

    importance_before = len(list_importance_paths(spider_dir))

    for _, row in selected.iterrows():
        community_id = row["community_id"]
        original_json_path = Path(row["community_json_path"])
        t0 = time.monotonic()

        count_before = len(list_importance_paths(spider_dir))
        used_reduced = False
        reduced_json_path = ""
        selected_seeds: List[str] = []
        execution_json = original_json_path

        try:
            execution_json, used_reduced, selected_seeds = prepare_execution_json(
                community_id=community_id,
                original_json_path=original_json_path,
                ce_labels=ce_labels,
                appr_seeds=appr_seeds,
                use_reduced_json=use_reduced_json,
                max_seeds_per_community=max_seeds_per_community,
                temp_dir=temp_dir,
                spider_dir=spider_dir,
            )
            reduced_json_path = str(execution_json) if used_reduced else ""
        except Exception as exc:
            runtime = time.monotonic() - t0
            run_rows.append(
                {
                    "community_id": community_id,
                    "used_reduced_json": used_reduced,
                    "reduced_json_path": reduced_json_path,
                    "selected_seed_count": len(selected_seeds),
                    "selected_seeds": ";".join(selected_seeds),
                    "before_importance_files": count_before,
                    "after_importance_files": count_before,
                    "new_importance_files": 0,
                    "status": "error",
                    "error_message": f"prepare json failed: {exc}",
                    "runtime_seconds": round(runtime, 2),
                }
            )
            logging.error("%s: prepare failed - %s", community_id, exc)
            continue

        seed_plan_logs.append(
            {
                "community_id": community_id,
                "selected_seeds": selected_seeds,
                "reduced_json_path": reduced_json_path,
            }
        )
        logging.info(
            "%s: selected %d seeds: %s",
            community_id,
            len(selected_seeds),
            ", ".join(selected_seeds) if selected_seeds else "(full json)",
        )

        status, _exit_code, error_message = run_analysis_for_community(
            execution_json, timeout_seconds
        )
        count_after = len(list_importance_paths(spider_dir))
        runtime = time.monotonic() - t0

        run_rows.append(
            {
                "community_id": community_id,
                "used_reduced_json": used_reduced,
                "reduced_json_path": reduced_json_path,
                "selected_seed_count": len(selected_seeds),
                "selected_seeds": ";".join(selected_seeds),
                "before_importance_files": count_before,
                "after_importance_files": count_after,
                "new_importance_files": count_after - count_before,
                "status": status,
                "error_message": error_message,
                "runtime_seconds": round(runtime, 2),
            }
        )

        logging.info(
            "%s: status=%s new_importance=%d runtime=%.1fs",
            community_id,
            status,
            count_after - count_before,
            runtime,
        )

        # Refresh appr seed cache so later communities prefer still-missing seeds.
        appr_seeds = list_appr_seed_addresses(spider_dir)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    importance_after = len(list_importance_paths(spider_dir))
    logging.info(
        "Batch complete: importance files %d -> %d (+%d)",
        importance_before,
        importance_after,
        importance_after - importance_before,
    )
    return pd.DataFrame(run_rows), seed_plan_logs


def generate_reduced_json_previews(
    plan: pd.DataFrame,
    ce_labels: Set[str],
    appr_seeds: Set[str],
    use_reduced_json: bool,
    max_seeds_per_community: int,
    temp_dir: Path,
    execute_one_first: bool,
) -> List[Dict[str, Any]]:
    """Dry-run helper: build reduced JSON previews for selected communities."""
    selected = plan[plan["selected"] == True].sort_values("selection_rank")  # noqa: E712
    if execute_one_first:
        selected = selected.head(1)

    previews: List[Dict[str, Any]] = []
    if not use_reduced_json:
        return previews

    for _, row in selected.iterrows():
        community_id = row["community_id"]
        original_json_path = Path(row["community_json_path"])
        selected_seeds = select_ce_seeds_for_community(
            community_json_path=original_json_path,
            ce_labels=ce_labels,
            appr_seeds=appr_seeds,
            max_seeds=max_seeds_per_community,
        )
        if not selected_seeds:
            logging.warning("%s: no CE seeds for reduced preview", community_id)
            continue
        original_data = load_community_json_raw(original_json_path)
        reduced_data = build_reduced_community_json(original_data, selected_seeds)
        reduced_path = write_reduced_json(reduced_data, temp_dir, community_id)
        previews.append(
            {
                "community_id": community_id,
                "selected_seeds": selected_seeds,
                "reduced_json_path": str(reduced_path),
            }
        )
        logging.info(
            "%s preview: %d seeds -> %s",
            community_id,
            len(selected_seeds),
            reduced_path,
        )
    return previews


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def write_log(
    log_path: Path,
    args: argparse.Namespace,
    inventory: pd.DataFrame,
    plan: pd.DataFrame,
    run_results: Optional[pd.DataFrame],
    seed_plan_logs: List[Dict[str, Any]],
    reduced_previews: List[Dict[str, Any]],
    appr_seed_count: int,
    ce_count: int,
    be_count: int,
) -> None:
    """Write human-readable experiment log."""
    selected = plan[plan["selected"] == True]  # noqa: E712
    lines = [
        "Phase-1 APPR Coverage Runner Log",
        "=" * 50,
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"project_root: {PROJECT_ROOT}",
        f"dry_run: {args.dry_run}",
        f"execute: {args.execute}",
        f"use_reduced_json: {args.use_reduced_json}",
        f"execute_one_first: {args.execute_one_first}",
        f"max_seeds_per_community: {args.max_seeds_per_community}",
        f"temp_dir: {args.temp_dir}",
        f"communities_dir: {args.communities_dir}",
        f"max_communities: {args.max_communities}",
        f"min_malicious_seeds: {args.min_malicious_seeds}",
        f"target_appr_per_community: {args.target_appr_per_community}",
        f"max_community_size: {args.max_community_size}",
        f"timeout_minutes: {args.timeout_minutes}",
        f"sleep_seconds: {args.sleep_seconds}",
        "",
        f"communities_scanned: {len(inventory)}",
        f"communities_selected: {len(selected)}",
        f"appr_importance_files: {len(list_importance_paths(PROJECT_ROOT / 'Spider'))}",
        f"unique_appr_seeds: {appr_seed_count}",
        f"ce_label_count: {ce_count}",
        f"be_label_count: {be_count}",
        "",
        "Selected communities:",
    ]

    if selected.empty:
        lines.append("  (none)")
    else:
        for _, row in selected.iterrows():
            lines.append(
                f"  #{int(row['selection_rank'])} {row['community_id']}: "
                f"malicious={int(row['malicious_seed_count'])}, "
                f"size={int(row['community_size'])}, "
                f"existing_appr={int(row['existing_appr_malicious_count'])}"
            )

    preview_logs = seed_plan_logs or reduced_previews
    if preview_logs:
        lines.extend(["", "Reduced JSON seed selection:"])
        for item in preview_logs:
            seeds = item.get("selected_seeds", [])
            seed_text = ", ".join(seeds) if isinstance(seeds, list) else str(seeds)
            lines.append(f"  - {item['community_id']}: seeds=[{seed_text}]")
            if item.get("reduced_json_path"):
                lines.append(f"    reduced_json: {item['reduced_json_path']}")

    if run_results is not None and not run_results.empty:
        lines.extend(["", "Execution results:"])
        for _, row in run_results.iterrows():
            lines.append(
                f"  - {row['community_id']}: {row['status']} "
                f"(+{int(row['new_importance_files'])} importance files, "
                f"{row['runtime_seconds']}s)"
            )
            if row.get("selected_seeds"):
                lines.append(f"    seeds: {row['selected_seeds']}")
        success = (run_results["status"] == "success").sum()
        lines.append(
            f"\nExecution summary: {success}/{len(run_results)} succeeded, "
            f"+{int(run_results['new_importance_files'].sum())} new importance files"
        )
    elif args.execute and not args.dry_run:
        lines.append("\nExecution results: (no communities executed)")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-1 APPR coverage runner: plan and optionally execute "
            "run_analysis.py to increase APPR importance coverage."
        )
    )
    parser.add_argument(
        "--communities_dir",
        default="results1/combined",
        help="Community JSON directory (default: results1/combined)",
    )
    parser.add_argument(
        "--ce_file",
        default="labels/CE.csv",
        help="Malicious label CSV (default: labels/CE.csv)",
    )
    parser.add_argument(
        "--be_file",
        default="labels/BE.csv",
        help="Benign label CSV (default: labels/BE.csv)",
    )
    parser.add_argument(
        "--max_communities",
        type=int,
        default=10,
        help="Maximum communities to select for execution (default: 10)",
    )
    parser.add_argument(
        "--min_malicious_seeds",
        type=int,
        default=5,
        help="Minimum malicious seeds per community (default: 5)",
    )
    parser.add_argument(
        "--target_appr_per_community",
        type=int,
        default=5,
        help="Select communities with fewer existing APPR files (default: 5)",
    )
    parser.add_argument(
        "--max_community_size",
        type=int,
        default=200,
        help="Maximum community size (default: 200)",
    )
    parser.add_argument(
        "--use_reduced_json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reduced CE-seed JSON instead of full community JSON (default: True)",
    )
    parser.add_argument(
        "--max_seeds_per_community",
        type=int,
        default=6,
        help="Maximum CE seeds per reduced community JSON (default: 6)",
    )
    parser.add_argument(
        "--temp_dir",
        default="phase1/appr_coverage/temp_communities",
        help="Directory for reduced community JSON files",
    )
    parser.add_argument(
        "--execute_one_first",
        action="store_true",
        help="Execute only the top-ranked selected community (selection_rank=1)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only generate coverage plan; do not execute run_analysis.py",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute run_analysis.py for selected communities",
    )
    parser.add_argument(
        "--sleep_seconds",
        type=int,
        default=5,
        help="Pause between run_analysis.py invocations (default: 5)",
    )
    parser.add_argument(
        "--timeout_minutes",
        type=int,
        default=10,
        help="Per-community timeout in minutes (default: 10)",
    )
    return parser.parse_args()


def resolve_path(relative: str) -> Path:
    """Resolve a project-relative path against PROJECT_ROOT."""
    path = Path(relative)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    communities_dir = resolve_path(args.communities_dir)
    ce_file = resolve_path(args.ce_file)
    be_file = resolve_path(args.be_file)
    spider_dir = PROJECT_ROOT / "Spider"
    output_dir = PROJECT_ROOT / "phase1" / "appr_coverage"
    temp_dir = resolve_path(args.temp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ce_labels = load_label_addresses(ce_file, "CE")
    be_labels = load_label_addresses(be_file, "BE")
    community_files = discover_community_files(communities_dir)
    appr_seeds = list_appr_seed_addresses(spider_dir)

    inventory = build_community_inventory(
        community_files=community_files,
        ce_labels=ce_labels,
        be_labels=be_labels,
        appr_seeds=appr_seeds,
    )
    plan = select_communities_for_run(
        inventory=inventory,
        min_malicious_seeds=args.min_malicious_seeds,
        max_community_size=args.max_community_size,
        target_appr_per_community=args.target_appr_per_community,
        max_communities=args.max_communities,
    )

    plan_path = output_dir / PLAN_CSV
    plan.to_csv(plan_path, index=False, encoding="utf-8")

    selected_count = int(plan["selected"].sum()) if not plan.empty else 0
    logging.info("Communities scanned: %d", len(inventory))
    logging.info("Communities selected: %d", selected_count)
    logging.info("Plan saved: %s", plan_path)

    run_results: Optional[pd.DataFrame] = None
    seed_plan_logs: List[Dict[str, Any]] = []
    reduced_previews: List[Dict[str, Any]] = []

    if args.dry_run and args.use_reduced_json:
        reduced_previews = generate_reduced_json_previews(
            plan=plan,
            ce_labels=ce_labels,
            appr_seeds=appr_seeds,
            use_reduced_json=True,
            max_seeds_per_community=args.max_seeds_per_community,
            temp_dir=temp_dir,
            execute_one_first=args.execute_one_first,
        )

    if args.execute and not args.dry_run:
        if selected_count == 0:
            logging.warning("No communities selected; skipping execution.")
        else:
            run_results, seed_plan_logs = execute_coverage_plan(
                plan=plan,
                spider_dir=spider_dir,
                ce_labels=ce_labels,
                appr_seeds=appr_seeds,
                use_reduced_json=args.use_reduced_json,
                max_seeds_per_community=args.max_seeds_per_community,
                temp_dir=temp_dir,
                sleep_seconds=args.sleep_seconds,
                timeout_minutes=args.timeout_minutes,
                execute_one_first=args.execute_one_first,
            )
            results_path = output_dir / RUN_RESULTS_CSV
            if run_results is not None and not run_results.empty:
                run_results.to_csv(results_path, index=False, encoding="utf-8")
                logging.info("Run results saved: %s", results_path)
            elif results_path.exists():
                logging.info("No new run rows; keeping existing %s", results_path)

    log_path = output_dir / LOG_TXT
    write_log(
        log_path=log_path,
        args=args,
        inventory=inventory,
        plan=plan,
        run_results=run_results,
        seed_plan_logs=seed_plan_logs,
        reduced_previews=reduced_previews,
        appr_seed_count=len(appr_seeds),
        ce_count=len(ce_labels),
        be_count=len(be_labels),
    )

    mode = "DRY RUN" if args.dry_run or not args.execute else "EXECUTE"
    print(f"\n[{mode}] Output directory: {output_dir.resolve()}")
    print(f"[{mode}] Communities scanned: {len(inventory)}")
    print(f"[{mode}] Communities selected: {selected_count}")
    print(f"[{mode}] Use reduced JSON: {args.use_reduced_json}")
    print(f"[{mode}] Plan CSV: {plan_path.resolve()}")
    print(f"[{mode}] Log: {log_path.resolve()}")
    if reduced_previews:
        print(f"[{mode}] Reduced JSON previews: {len(reduced_previews)}")
    if run_results is not None:
        print(f"[{mode}] Run results: {(output_dir / RUN_RESULTS_CSV).resolve()}")


if __name__ == "__main__":
    main()
