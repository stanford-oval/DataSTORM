"""Assemble the published reproducibility artifacts into ``results/``.

Source data lives outside this repo (the working tree that produced the runs,
and the InsightBench harness clone). This script copies the publishable subset
in, leaving behind two categories of material:

* **Bulk exploration state** -- ``tree.json`` (130 MB across 20 ACLED topics),
  ``url_to_info.json`` and ``staged_report_provenance.json``. Too large for git,
  and the ACLED trees embed verbatim source records (see LICENSING below).
* **Third-party ground truth** -- InsightBench's reference insights and
  summaries. Scores and our predictions are ours to publish; their labels are
  not, so they are stripped.

LICENSING: ACLED event records, including the ``notes`` narrative field, appear
inside the exploration trees. The final reports do not contain them -- verified
by scanning all 42 report files for event IDs, ``notes`` columns and data
tables. That is why reports ship and trees do not.

Usage:
    python scripts/export_results.py --acled-src /path/to/datatalk/acled \\
                                     --insightbench-src /path/to/insight-bench
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-topic files small enough and clean enough to publish.
ACLED_REPORT_FILES = [
    "co_storm_report.txt",
    "co_storm_report_staged.md",
    "co_storm_report_revised.txt",
]
ACLED_METADATA_FILES = [
    "run_metadata.json",
    "fact_check_summary.json",
    "staged_report_title.json",
    "staged_report_plan.json",
    "staged_report_sections.json",
    "article_critique.json",
    "input.txt",
    "storm_gen_outline.txt",
    "direct_gen_outline.txt",
]
# Deliberately excluded: tree.json, url_to_info.json,
# staged_report_provenance.json, staged_report_notes.json,
# conversation_log.json, raw_search_results.json, warmstart_conversation.json.

# InsightBench prediction fields that are the benchmark's, not ours.
INSIGHTBENCH_DROP = {"ground_truth_insights", "ground_truth_summary"}

EVENT_ID = re.compile(r"\b[A-Z]{3}\d{4,}\b")

# Local filesystem paths recorded in run metadata carry no value to a reader and
# leak the layout of the machine that produced the runs. Citation URLs are left
# alone: they are the provenance links the reports depend on.
PATH_SCRUBS = [
    (re.compile(r"/home/[A-Za-z0-9_.-]+/storm-main/"), "<repo>/"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/storm/"), "<repo>/"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/"), "<local>/"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/"), "<local>/"),
    (re.compile(r"/data1/"), "<data>/"),
    (re.compile(r"/datadrive/"), "<data>/"),
]


def scrub_paths(text: str) -> str:
    for pattern, replacement in PATH_SCRUBS:
        text = pattern.sub(replacement, text)
    return text


def copy_scrubbed(src: Path, dest: Path) -> None:
    """Copy a text artifact, rewriting machine-local paths."""
    dest.write_text(
        scrub_paths(src.read_text(encoding="utf-8", errors="ignore")), encoding="utf-8"
    )


def tracked_runs(src: Path) -> dict[str, str]:
    """Map topic -> the run directory committed to git.

    The working tree that produced these runs accumulates dozens of local
    experiment directories per topic; only one per topic is the published
    baseline, and that is the one under version control. Picking by timestamp
    silently exports the wrong run, so selection is by git tracking instead.
    """
    try:
        # -z gives raw NUL-separated paths; without it git quote-escapes any
        # path containing non-ASCII (several topic titles use a curly
        # apostrophe), which silently breaks parsing for those topics.
        listing = subprocess.run(
            ["git", "-C", str(src), "ls-files", "-z", "--", "."],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}

    mapping: dict[str, str] = {}
    for line in listing.split("\0"):
        if not line:
            continue
        parts = line.split("/")
        if len(parts) < 3:
            continue
        topic, run = parts[0], parts[1]
        mapping.setdefault(topic, run)
    return mapping


def export_acled(src: Path, out: Path) -> dict:
    reports_dir = out / "reports"
    meta_dir = out / "metadata"
    reports_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    published = tracked_runs(src)
    if not published:
        raise SystemExit(
            f"{src} is not a git checkout, so the published run for each topic "
            f"cannot be identified. Export from a clean checkout of the results "
            f"repo, where exactly one run per topic is tracked."
        )

    topics, copied, flagged, skipped = 0, 0, [], []
    for topic_dir in sorted(src.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if not topic_dir.is_dir():
            continue
        run_name = published.get(topic_dir.name)
        if run_name is None:
            skipped.append(topic_dir.name)
            continue
        run = topic_dir / run_name
        if not run.is_dir():
            skipped.append(topic_dir.name)
            continue
        topics += 1

        for name in ACLED_REPORT_FILES:
            srcf = run / name
            if not srcf.exists():
                continue
            text = srcf.read_text(encoding="utf-8", errors="ignore")
            # Guard the licensing claim rather than trusting it.
            if EVENT_ID.search(text) or "| notes" in text:
                flagged.append(f"{topic_dir.name}/{name}")
            copy_scrubbed(srcf, reports_dir / f"{topic_dir.name}__{name}")
            copied += 1

        for name in ACLED_METADATA_FILES:
            srcf = run / name
            if srcf.exists():
                copy_scrubbed(srcf, meta_dir / f"{topic_dir.name}__{name}")
                copied += 1

    return {
        "topics": topics,
        "files": copied,
        "flagged_for_raw_data": flagged,
        "skipped_topics": skipped,
        "runs": {t: published[t] for t in sorted(published)},
    }


# Table 2 (reference-induced criteria matching + RACE) and Table 4a
# (leave-one-out ablations). Copied verbatim, only path-scrubbed.
ACLED_BENCHMARK_SOURCES = {
    # Paths are relative to the evaluations directory passed in.
    "race": (".", ("_costorm.json", "_openai_dr.json", "_openai_dr_csv.json")),
    "criteria_match": ("criteria_match", ("_criteria_match_",)),
}


def export_acled_benchmark(dr_root: Path, ablations_root: Path, out: Path) -> dict:
    """Copy the evaluation outputs behind Tables 2 and 4a."""
    counts = {}
    for name, (subdir, markers) in ACLED_BENCHMARK_SOURCES.items():
        dest = out / name
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for src in sorted((dr_root / subdir).glob("*.json")):
            if not any(m in src.name for m in markers):
                continue
            if "remove_unused_tables" in src.name:
                continue
            copy_scrubbed(src, dest / src.name)
            n += 1
        counts[name] = n

    if ablations_root.exists():
        n = 0
        for src in sorted(ablations_root.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(ablations_root)
            dest = out / "ablations" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            copy_scrubbed(src, dest)
            n += 1
        counts["ablations"] = n
    return counts


def export_insightbench(run_dir: Path, out: Path, scores_subdir: str = "scores") -> dict:
    scores_dir = out / scores_subdir
    scores_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, list[float]] = {}
    n = 0
    for path in sorted(glob.glob(str(run_dir / "predictions" / "pred_gt_*.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        n += 1
        clean = {k: v for k, v in payload.items() if k not in INSIGHTBENCH_DROP}
        for key, value in clean.items():
            if key.startswith("score_") and isinstance(value, (int, float)):
                metrics.setdefault(key, []).append(float(value))
        (scores_dir / os.path.basename(path)).write_text(
            scrub_paths(json.dumps(clean, indent=2, ensure_ascii=False)) + "\n",
            encoding="utf-8",
        )

    aggregates = {
        key: {
            "mean": round(statistics.mean(vals), 4),
            "n": len(vals),
        }
        for key, vals in sorted(metrics.items())
    }
    return {"datasets": n, "aggregates": aggregates}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acled-src", default=os.getenv("DATASTORM_ACLED_RUNS", ""))
    # The published run: 20260325_130410_benchmark_type-full_model_name-gpt-5
    # _eval_model_name-gpt-4o_notes- inside the InsightBench harness results dir.
    ap.add_argument(
        "--insightbench-run",
        default=os.getenv("INSIGHT_BENCH_RUN", ""),
    )
    # The InsightBench agent baseline, published for comparison:
    # baseline-20260331-034919 in the same results dir.
    ap.add_argument(
        "--insightbench-baseline",
        default=os.getenv("INSIGHT_BENCH_BASELINE", ""),
    )
    ap.add_argument("--acled-dr-evaluations", default=os.getenv("ACLED_DR_EVALUATIONS", ""),
                    help="openai_dr_baseline/evaluations (Table 2)")
    ap.add_argument("--acled-ablations", default=os.getenv("ACLED_ABLATIONS", ""),
                    help="eval_results/acled_ablations (Table 4a)")
    ap.add_argument("--out", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    out = Path(args.out)
    summary = {}

    acled_src = Path(args.acled_src)
    if acled_src.exists():
        summary["acled"] = export_acled(acled_src, out / "acled")
        print(f"ACLED: {summary['acled']['topics']} topics, "
              f"{summary['acled']['files']} files")
        if summary["acled"]["flagged_for_raw_data"]:
            print("  WARNING - reports containing raw event records:")
            for f in summary["acled"]["flagged_for_raw_data"]:
                print("   ", f)
    else:
        print(f"ACLED source not found: {acled_src}")

    if args.acled_dr_evaluations and Path(args.acled_dr_evaluations).exists():
        summary["acled_benchmark"] = export_acled_benchmark(
            Path(args.acled_dr_evaluations), Path(args.acled_ablations or "/nonexistent"),
            out / "acled_benchmark")
        print(f"ACLED benchmark: {summary['acled_benchmark']}")

    ib_run = Path(args.insightbench_run)
    if ib_run.exists():
        summary["insight_bench"] = export_insightbench(ib_run, out / "insight_bench")
        print(f"InsightBench: {summary['insight_bench']['datasets']} datasets")
        for k, v in summary["insight_bench"]["aggregates"].items():
            print(f"    {k:<56} {v['mean']:.4f}  (n={v['n']})")
    else:
        print(f"InsightBench run not found: {ib_run}")

    ib_base = Path(args.insightbench_baseline)
    if args.insightbench_baseline and ib_base.exists():
        summary["insight_bench_baseline"] = export_insightbench(
            ib_base, out / "insight_bench", scores_subdir="baseline_scores"
        )
        print(
            f"InsightBench baseline: "
            f"{summary['insight_bench_baseline']['datasets']} datasets"
        )
        for k, v in summary["insight_bench_baseline"]["aggregates"].items():
            print(f"    {k:<56} {v['mean']:.4f}  (n={v['n']})")
    elif args.insightbench_baseline:
        print(f"InsightBench baseline not found: {ib_base}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "MANIFEST.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
