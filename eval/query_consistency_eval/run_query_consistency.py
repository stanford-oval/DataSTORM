#!/usr/bin/env python3
"""
Batch driver for the query-consistency evaluator across report sets.

For each (condition, id) it locates the run directory that actually has
url_to_info.json (the SQL sources live there), runs the Tier-1 LLM judge, writes
per-report JSON under query_consistency_eval/results/<cond>/<id>.json, and
aggregates a per-condition summary.

Usage:
  # the no-QC arm (all ten topics)
  python query_consistency_eval/run_query_consistency.py --set no_qc --ids 1 2 4 8 9 10 11 12 17 18
  # aggregate only
  python query_consistency_eval/run_query_consistency.py --set no_qc --ids 1 2 4 8 9 10 11 12 17 18 --aggregate-only
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "query_consistency_eval"))

from query_consistency_evaluator import evaluate, DIMENSIONS  # noqa: E402

OUT_ROOT = REPO / "query_consistency_eval" / "results"


def find_run_dir(set_name: str, eval_id: int) -> Optional[Path]:
    if set_name == "baseline":
        # Prefer the run referenced by the baseline eval; else latest with url_to_info.
        ev = REPO / "openai_dr_baseline" / "evaluations" / f"{eval_id}.json"
        if ev.exists():
            gen = json.loads(ev.read_text()).get("costorm", {}).get("generated_file", "")
            d = Path(gen).parent
            if (d / "url_to_info.json").exists():
                return d
        base = REPO / "datatalk" / "acled" / str(eval_id)
    else:
        # Six of the ten no-QC runs were salvaged by re-summarising into
        # acled_ablations/no_qc_fixed/. Prefer the salvaged run when one exists
        # so `--set no_qc` regenerates the full ten-topic arm in one pass.
        root = REPO / "datatalk" / "acled_ablations"
        base = root / set_name / str(eval_id)
        if set_name == "no_qc":
            fixed = root / "no_qc_fixed" / str(eval_id)
            if fixed.exists():
                base = fixed
    if not base.exists():
        return None
    dirs = sorted((p.parent for p in base.glob("*/url_to_info.json")), key=lambda p: p.name)
    return dirs[-1] if dirs else None


def run_one(task) -> Dict[str, Any]:
    set_name, eval_id, run_dir, model = task
    out_dir = OUT_ROOT / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = evaluate(run_dir, model)
    except Exception as exc:  # pragma: no cover
        result = {"report_dir": str(run_dir), "error": str(exc), "applicable": False, "overall_score": None}
    result["id"] = eval_id
    (out_dir / f"{eval_id}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    sc = result.get("overall_score")
    nq = result.get("num_queries", "?")
    print(f"  [{set_name}/{eval_id}] {nq} queries -> overall={round(sc,3) if isinstance(sc,(int,float)) else sc}", flush=True)
    return result


def aggregate(set_name: str, ids: List[int]) -> None:
    out_dir = OUT_ROOT / set_name
    rows = []
    for i in ids:
        p = out_dir / f"{i}.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
    applicable = [r for r in rows if r.get("applicable") and r.get("overall_score") is not None]

    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return sum(xs) / len(xs) if xs else None

    dim_means = {}
    for d in DIMENSIONS:
        vals = []
        for r in applicable:
            for ds in r.get("per_dimension", []):
                if ds.get("dimension") == d:
                    vals.append(ds.get("score"))
        dim_means[d] = mean(vals)

    n_conflicts = sum(len(r.get("conflicts", [])) for r in applicable)
    n_result_changing = sum(1 for r in applicable for c in r.get("conflicts", []) if c.get("severity") == "result-changing")

    print(f"\n==================== QUERY CONSISTENCY: {set_name} ====================")
    print(f"reports evaluated (>=2 SQL): {len(applicable)} / {len(rows)}  "
          f"(N/A — <2 queries: {len(rows) - len(applicable)})")
    ov = mean([r['overall_score'] for r in applicable])
    print(f"mean overall consistency: {round(ov,3) if ov is not None else '—'}")
    print("per-dimension means:")
    for d in DIMENSIONS:
        v = dim_means[d]
        print(f"  {d:<22} {round(v,3) if v is not None else '—'}")
    print(f"total conflicts: {n_conflicts}  (result-changing: {n_result_changing})")
    print("per-report:")
    for r in rows:
        sc = r.get("overall_score")
        print(f"  id={r.get('id')}: {r.get('num_queries')} queries, "
              f"overall={round(sc,3) if isinstance(sc,(int,float)) else 'N/A'}, "
              f"conflicts={len(r.get('conflicts', []))}")
    summary = {
        "set": set_name, "ids": ids, "n_applicable": len(applicable),
        "mean_overall": ov, "per_dimension_means": dim_means,
        "total_conflicts": n_conflicts, "result_changing_conflicts": n_result_changing,
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if not args.aggregate_only:
        tasks = []
        for i in args.ids:
            rd = find_run_dir(args.set, i)
            if rd is None:
                print(f"  [{args.set}/{i}] no run dir with url_to_info.json — skipping", flush=True)
                continue
            tasks.append((args.set, i, rd, args.model))
        print(f"[qc] {len(tasks)} report(s) to evaluate for set={args.set}\n", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(run_one, tasks))
    aggregate(args.set, args.ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
