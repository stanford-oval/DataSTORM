#!/usr/bin/env python3
"""
Compare an ablation set's paper-metrics (criteria-match, RACE, DB-use) against the
full-system baseline, on the topics where both have results.

baseline ("costorm") scores come from the already-computed paper eval outputs:
  openai_dr_baseline/evaluations/criteria_match/<id>.json  -> criteria_match.costorm.overall_score
  openai_dr_baseline/evaluations/<id>.json                 -> costorm.scores (RACE)
  breakdown_evaluator/data_source_evaluations/<id>_costorm.json -> pct_acled
ablation scores come from the eval_ablation_metrics.py driver outputs:
  eval_results/acled_ablations/<set>/<id>_{criteria,race,datasource}.json

Usage: python acled_ablation/compare_to_baseline.py --set no_inductive
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IDS = [1, 2, 4, 8, 9, 10, 11, 12, 17, 18]


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def baseline_scores(i):
    out = {}
    cm = _load(REPO / "openai_dr_baseline" / "evaluations" / "criteria_match" / f"{i}.json")
    if cm and "costorm" in cm.get("criteria_match", {}):
        out["criteria"] = cm["criteria_match"]["costorm"].get("overall_score")
    ev = _load(REPO / "openai_dr_baseline" / "evaluations" / f"{i}.json")
    if ev and isinstance(ev.get("costorm"), dict):
        s = ev["costorm"].get("scores", {})
        out["race"] = {k: s.get(k) for k in ("overall_score", "comprehensiveness", "insight", "instruction_following", "readability")}
    ds = _load(REPO / "breakdown_evaluator" / "data_source_evaluations" / f"{i}_costorm.json")
    if ds:
        out["db"] = ds.get("pct_acled")
    return out


RACE_KEYS = ("overall_score", "comprehensiveness", "insight", "instruction_following", "readability")


def ablation_scores(set_name, i, fill_zero=False):
    base = REPO / "eval_results" / "acled_ablations" / set_name
    out = {}
    c = _load(base / f"{i}_criteria.json")
    if c:
        out["criteria"] = c.get("overall_score")
    elif fill_zero:
        out["criteria"] = 0.0
    r = _load(base / f"{i}_race.json")
    if r:
        out["race"] = {k: r.get(k) for k in RACE_KEYS}
    elif fill_zero:
        out["race"] = {k: 0.0 for k in RACE_KEYS}
    d = _load(base / f"{i}_datasource.json")
    if d:
        out["db"] = d.get("pct_acled")
    elif fill_zero:
        out["db"] = 0.0
    return out


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def fmt(v, pct=False):
    if v is None:
        return "  —  "
    return f"{v*100:5.1f}" + ("%" if pct else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--fill-zero", action="store_true",
                    help="include ALL topics; substitute 0 for content-blocked / missing ablation evals")
    ap.add_argument("--ids", type=int, nargs="+", default=None,
                    help="restrict to this explicit topic-id set (for a common-N comparison across conditions)")
    args = ap.parse_args()

    ids = args.ids if args.ids else IDS
    rows = []
    for i in ids:
        b, a = baseline_scores(i), ablation_scores(args.set, i, fill_zero=args.fill_zero)
        rows.append((i, b, a))

    print(f"\n========== BASELINE vs {args.set.upper()} (paper metrics) ==========")
    print(f"{'id':>3} | {'AvgMatch (b/a)':>15} | {'RACE (b/a)':>13} | {'DB-use (b/a)':>15}")
    print("-" * 60)
    pair_ids = []
    for i, b, a in rows:
        bc, ac = b.get("criteria"), a.get("criteria")
        bro = (b.get("race") or {}).get("overall_score"); aro = (a.get("race") or {}).get("overall_score")
        bd, ad = b.get("db"), a.get("db")
        ready = ac is not None and aro is not None and ad is not None
        if ready:
            pair_ids.append(i)
        tag = "" if ready else "  (ablation incomplete)"
        print(f"{i:>3} | {fmt(bc,1)}/{fmt(ac,1)} | {fmt(bro)}/{fmt(aro)} | {fmt(bd,1)}/{fmt(ad,1)}{tag}")

    print("-" * 60)
    print(f"topics with complete ablation metrics: {pair_ids} (N={len(pair_ids)})")
    def col(side, metric, sub=None):
        vals = []
        for i, b, a in rows:
            if i not in pair_ids:
                continue
            src = (b if side == "b" else a)
            v = (src.get("race") or {}).get(sub) if metric == "race" else src.get(metric)
            vals.append(v)
        return mean(vals)
    print(f"\nMEAN over N={len(pair_ids)}:")
    print(f"  AvgMatch:  baseline {fmt(col('b','criteria'),1)}  vs  {args.set} {fmt(col('a','criteria'),1)}")
    print(f"  RACE Overall: baseline {fmt(col('b','race','overall_score'))}  vs  {args.set} {fmt(col('a','race','overall_score'))}")
    for d in ("comprehensiveness", "insight", "instruction_following", "readability"):
        print(f"     {d:<20} baseline {fmt(col('b','race',d))}  vs  {args.set} {fmt(col('a','race',d))}")
    print(f"  DB-use:    baseline {fmt(col('b','db'),1)}  vs  {args.set} {fmt(col('a','db'),1)}")
    print("(AvgMatch & DB-use are %; RACE 0-100, 50=parity. 'insight'=Depth.)")


if __name__ == "__main__":
    main()
