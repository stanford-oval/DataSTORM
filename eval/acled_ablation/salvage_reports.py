#!/usr/bin/env python3
"""
Salvage final reports for runs whose knowledge curation completed but whose
report generation failed — without re-running the (expensive) curation.

This is primarily for the `no_thesis` ablation: those runs crashed because the
staged report generator required a thesis (now fixed via allow_empty_thesis), so
they have tree.json / warmstart_conversation.json on disk but no report. This
script regenerates, per run dir:
  * co_storm_report_staged.md  (+ staged_report_*.json intermediates)
  * co_storm_report.txt
  * url_to_info.json            (reconstructed from tree + warmstart evidence)

It reuses the same disk-assembly helpers the live pipeline uses, so the output
matches a normal run. Run from the repo root with the `storm` conda env.

Usage:
  # one run dir (verify first)
  python acled_ablation/salvage_reports.py --run-dir datatalk/acled_ablations/no_thesis/18/<run>
  # every no_thesis run dir missing a report
  python acled_ablation/salvage_reports.py --ablation-root datatalk/acled_ablations/no_thesis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from knowledge_storm.utils import load_api_key  # noqa: E402

load_api_key(toml_file_path=str(REPO_ROOT / "secrets.toml"))

from final_report_gen_utils import (  # noqa: E402
    generate_report_from_data,
    SourceRegistry,
    collect_tree_evidence,
    load_warmstart_evidence_from_disk,
)


def _db_description(domain: str = "acled") -> str:
    """Use the same DB description the live pipeline uses, for fidelity."""
    try:
        from datatalk_basic_article_generation_costorm import db_description_mapping
        if domain in db_description_mapping:
            return db_description_mapping[domain]
    except Exception as exc:  # pragma: no cover
        print(f"[salvage] could not import db_description_mapping: {exc}", file=sys.stderr)
    return "You have access to a relevant ACLED conflict-events database."


def reconstruct_url_to_info(tree: Dict[str, Any], warmstart_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rebuild url_to_info.json from the evidence the report can cite.

    Mirrors the live pipeline's ``all_search_results`` assembly: warmstart
    cited_info, then each is_final_selected tree node's first search result.
    Each search-result dict already carries the full meta (SQL, sql_result,
    csv_path, designation, query), which is what the viewer needs.
    """
    out: Dict[str, Any] = {"url_to_unified_index": {}, "url_to_info": {}}
    counter = 0

    def add(info: Dict[str, Any]) -> None:
        nonlocal counter
        if not isinstance(info, dict):
            return
        url = info.get("url") or ""
        if not url or url in out["url_to_info"]:
            return
        counter += 1
        out["url_to_unified_index"][url] = counter
        info = dict(info)
        info["citation_uuid"] = counter
        out["url_to_info"][url] = info

    for turn in (warmstart_data or []):
        for info in (turn.get("cited_info") or []):
            add(info)

    def walk(node: Dict[str, Any]) -> None:
        if node.get("is_final_selected"):
            sr = ((node.get("dlg_turn") or {}).get("search_results")) or []
            if sr:
                add(sr[0])
        for child in (node.get("children") or []):
            walk(child)

    walk(tree)
    return out


def salvage_run(run_dir: Path, *, model: str = "gpt-5", langfuse_readonly: bool = True) -> bool:
    run_dir = run_dir.resolve()
    tree_path = run_dir / "tree.json"
    warmstart_path = run_dir / "warmstart_conversation.json"
    input_path = run_dir / "input.txt"

    if not tree_path.exists():
        print(f"[salvage] SKIP (no tree.json): {run_dir}")
        return False
    if (run_dir / "co_storm_report.txt").exists():
        print(f"[salvage] SKIP (already has report): {run_dir}")
        return False

    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    warmstart_data = json.loads(warmstart_path.read_text(encoding="utf-8")) if warmstart_path.exists() else []
    topic = input_path.read_text(encoding="utf-8").strip() if input_path.exists() else ""
    if not topic:
        print(f"[salvage] SKIP (no topic/input.txt): {run_dir}")
        return False

    # Use the tree's thesis if present (no_qc etc.); empty for no_thesis runs,
    # in which case generation proceeds thesis-free via allow_empty_thesis.
    thesis = (tree.get("thesis") or "").strip()

    # Assemble evidence exactly like the disk-loader path: warmstart records
    # first, then tree-selected evidence re-indexed to follow them.
    registry = SourceRegistry()
    warmstart_records = load_warmstart_evidence_from_disk(warmstart_data, registry, evidence_id_start=1)
    tree_records = collect_tree_evidence(tree, registry)
    for i, rec in enumerate(tree_records):
        rec.evidence_id = len(warmstart_records) + i + 1
    evidence_records = warmstart_records + tree_records

    print(f"[salvage] {run_dir.name}")
    print(f"  topic: {topic[:80]}")
    print(f"  evidence: {len(warmstart_records)} warmstart + {len(tree_records)} tree = {len(evidence_records)}")
    if not evidence_records:
        print("  ERROR: no evidence assembled (no is_final_selected nodes?) — cannot generate.")
        return False

    result = generate_report_from_data(
        topic=topic,
        thesis=thesis,  # "" for no_thesis runs
        tree=tree,
        warmstart_data=warmstart_data,
        run_dir=str(run_dir),
        db_description=_db_description("acled"),
        model=model,
        evidence_records=evidence_records,
        registry=registry,
        langfuse_readonly=langfuse_readonly,
        allow_empty_thesis=(not thesis),
    )

    report = (result or {}).get("report_content") or ""
    if not report:
        print("  ERROR: generation returned empty report_content.")
        return False

    (run_dir / "co_storm_report.txt").write_text(report, encoding="utf-8")
    url_to_info = reconstruct_url_to_info(tree, warmstart_data)
    (run_dir / "url_to_info.json").write_text(
        json.dumps(url_to_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fc = (result or {}).get("fact_check_stats") or {}
    print(f"  WROTE co_storm_report.txt ({len(report)} chars), "
          f"url_to_info.json ({len(url_to_info['url_to_info'])} sources), "
          f"staged.md + intermediates")
    if fc:
        print(f"  fact_check: {fc.get('issues_found', 0)}/{fc.get('total_checked', 0)} issue(s)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="A single run directory to salvage.")
    g.add_argument("--ablation-root", help="Root (e.g. datatalk/acled_ablations/no_thesis); salvages every <id>/<run> missing a report.")
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--langfuse-readonly", action="store_true", default=True)
    args = ap.parse_args()

    if args.run_dir:
        ok = salvage_run(Path(args.run_dir), model=args.model)
        return 0 if ok else 1

    root = Path(args.ablation_root)
    run_dirs = sorted(p.parent for p in root.glob("*/*/tree.json"))
    print(f"[salvage] {len(run_dirs)} run dir(s) under {root}\n")
    done = 0
    for rd in run_dirs:
        try:
            if salvage_run(rd, model=args.model):
                done += 1
        except Exception as exc:
            import traceback
            print(f"[salvage] FAILED {rd}: {exc}\n{traceback.format_exc()}")
        print()
    print(f"[salvage] salvaged {done}/{len(run_dirs)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
