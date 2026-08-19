#!/usr/bin/env python3
"""
Run the three ACLED automatic metrics (paper section 4.2.1) on ablation report
sets and aggregate a paper-style summary table.

Metrics (all judged by gpt-5 via the existing single-report evaluators):
  1. criteria   — reference-induced criteria matching  -> overall_score (0-1)
                  openai_dr_baseline/run_criteria_match_evaluations.py
  2. race       — RACE framework (overall + 4 dims)     -> *_score (0-1, .5=parity)
                  deep_research_bench_evaluator/evaluator.py
  3. datasource — database-use ratio                    -> pct_acled (0-1)
                  breakdown_evaluator/data_source_evaluator.py

Per (set, id) we locate datatalk/acled_ablations/<set>/<id>/<run>/co_storm_report.txt,
pull the prompt + reference for that id from acled_evaluations.json, and run each
metric, writing per-eval JSON under eval_results/acled_ablations/<set>/<id>_<metric>.json.

Usage:
  python acled_ablation/eval_ablation_metrics.py --sets no_qc no_thesis
  python acled_ablation/eval_ablation_metrics.py --sets no_qc no_thesis --aggregate-only
Run from repo root with the storm conda env.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
EVALS_JSON = REPO / "openai_dr_baseline" / "acled_evaluations.json"
ABLATION_ROOT = REPO / "datatalk" / "acled_ablations"
OUT_ROOT = REPO / "eval_results" / "acled_ablations"

METRICS = {
    "criteria": {
        "script": REPO / "openai_dr_baseline" / "run_criteria_match_evaluations.py",
        "needs_reference": True,
    },
    "race": {
        "script": REPO / "deep_research_bench_evaluator" / "evaluator.py",
        "needs_reference": True,
    },
    "datasource": {
        "script": REPO / "breakdown_evaluator" / "data_source_evaluator.py",
        "needs_reference": False,
    },
}


def load_eval_specs() -> Dict[int, Dict[str, Any]]:
    data = json.loads(EVALS_JSON.read_text(encoding="utf-8"))
    return {int(e["id"]): e for e in data}


def find_report(set_name: str, eval_id: int) -> Optional[Path]:
    base = ABLATION_ROOT / set_name / str(eval_id)
    if not base.exists():
        return None
    reports = sorted(base.glob("*/co_storm_report.txt"))
    return reports[-1] if reports else None


def build_tasks(sets: List[str], specs: Dict[int, Dict[str, Any]], resume: bool):
    tasks = []
    for set_name in sets:
        out_dir = OUT_ROOT / set_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for eval_id, spec in sorted(specs.items()):
            report = find_report(set_name, eval_id)
            if report is None:
                continue
            for metric, cfg in METRICS.items():
                out_file = out_dir / f"{eval_id}_{metric}.json"
                if resume and out_file.exists():
                    continue
                tasks.append((set_name, eval_id, metric, cfg, spec, report, out_file))
    return tasks


def run_task(task) -> Dict[str, Any]:
    set_name, eval_id, metric, cfg, spec, report, out_file = task
    cmd = [
        PY, str(cfg["script"]),
        "--prompt", spec["prompt"],
        "--generated_file", str(report),
        "--output_file", str(out_file),
    ]
    if cfg["needs_reference"]:
        cmd += ["--reference_file", spec["reference_file"]]
    env = dict(os.environ)
    env["PYTHONWARNINGS"] = "ignore:Pydantic serializer warnings:UserWarning"
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=1800
        )
        ok = proc.returncode == 0 and out_file.exists()
        msg = "ok" if ok else (proc.stderr.strip().splitlines()[-1:] or ["no output"])[0]
    except subprocess.TimeoutExpired:
        ok, msg = False, "timeout"
    label = f"{set_name}/{eval_id}/{metric}"
    print(f"  [{'OK ' if ok else 'ERR'}] {label}: {msg}", flush=True)
    return {"set": set_name, "id": eval_id, "metric": metric, "ok": ok}


def aggregate(sets: List[str]) -> None:
    def avg(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else None

    rows = []
    for set_name in sets:
        out_dir = OUT_ROOT / set_name
        crit, overall, comp, depth, inst, read, dbuse = [], [], [], [], [], [], []
        ids = set()
        for f in out_dir.glob("*_criteria.json"):
            d = json.loads(f.read_text()); crit.append(d.get("overall_score")); ids.add(f.stem.split("_")[0])
        for f in out_dir.glob("*_race.json"):
            d = json.loads(f.read_text())
            overall.append(d.get("overall_score")); comp.append(d.get("comprehensiveness"))
            depth.append(d.get("insight")); inst.append(d.get("instruction_following"))
            read.append(d.get("readability")); ids.add(f.stem.split("_")[0])
        for f in out_dir.glob("*_datasource.json"):
            d = json.loads(f.read_text()); dbuse.append(d.get("pct_acled")); ids.add(f.stem.split("_")[0])
        rows.append({
            "set": set_name, "n": len(ids),
            "AvgMatch%": _pct(avg(crit)), "RACE_Overall": _x100(avg(overall)),
            "Comp": _x100(avg(comp)), "Depth": _x100(avg(depth)),
            "Inst": _x100(avg(inst)), "Read": _x100(avg(read)),
            "DBUse%": _pct(avg(dbuse)),
        })

    print("\n==================== ABLATION AUTOMATIC METRICS ====================")
    hdr = f"{'set':<11} {'n':>2}  {'AvgMatch':>8}  {'RACE':>6} {'Comp':>5} {'Depth':>5} {'Inst':>5} {'Read':>5}  {'DBUse':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['set']:<11} {r['n']:>2}  {r['AvgMatch%']:>7}%  {r['RACE_Overall']:>6} "
              f"{r['Comp']:>5} {r['Depth']:>5} {r['Inst']:>5} {r['Read']:>5}  {r['DBUse%']:>5}%")
    print("\n(AvgMatch & DBUse are %; RACE/dims on 0-100, 50=parity with reference. Depth=RACE 'insight'.)")
    (OUT_ROOT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_ROOT / 'summary.json'}")


def _x100(v):
    return round(v * 100, 1) if v is not None else "—"


def _pct(v):
    return round(v * 100, 1) if v is not None else "—"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", default=["no_qc", "no_thesis"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-resume", action="store_true", help="Recompute even if output exists.")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate(args.sets)
        return 0

    specs = load_eval_specs()
    tasks = build_tasks(args.sets, specs, resume=not args.no_resume)
    print(f"[eval] {len(tasks)} eval task(s) across sets={args.sets}, workers={args.workers}\n", flush=True)
    if tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(run_task, tasks))
        ok = sum(1 for r in results if r["ok"])
        print(f"\n[eval] {ok}/{len(results)} task(s) succeeded")
    aggregate(args.sets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
