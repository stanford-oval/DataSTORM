#!/usr/bin/env python3
"""
Data Source Evaluation Script

This script evaluates whether the insights in generated articles are derived from
ACLED data or other sources. It breaks down the generated article into atomic insights,
then classifies each insight as ACLED-sourced or not.

Usage:
    python run_data_source_evaluations.py                    # Run all evaluations
    python run_data_source_evaluations.py --ids 1 2         # Run specific evaluation IDs
    python run_data_source_evaluations.py --skip-costorm    # Only evaluate OpenAI outputs
    python run_data_source_evaluations.py --skip-openai     # Only evaluate CoStorm outputs
    python run_data_source_evaluations.py --resume          # Skip already completed evaluations
"""

import json
import os
import pathlib
import csv
import argparse
import glob
import subprocess
import shlex
from datetime import datetime

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STORM_ROOT = os.path.dirname(SCRIPT_DIR)
OPENAI_DR_BASELINE_DIR = os.path.join(STORM_ROOT, "openai_dr_baseline")
EVALUATIONS_FILE = os.path.join(OPENAI_DR_BASELINE_DIR, "acled_evaluations.json")
RUNS_DIR = os.path.join(OPENAI_DR_BASELINE_DIR, "runs")
EVALUATIONS_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data_source_evaluations")
COSTORM_OUTPUT_DIR = os.environ.get(
    "DATASTORM_ACLED_RUNS",
    str(pathlib.Path(__file__).resolve().parents[2] / "results" / "acled" / "runs"),
)

DATA_SOURCE_EVALUATOR_SCRIPT = os.path.join(SCRIPT_DIR, "data_source_evaluator.py")

# Conda environment name
CONDA_ENV = "storm"


def build_conda_shell_cmd(script: str, *args: str) -> str:
    """Build a shell command string that runs a Python script in the storm conda environment."""
    python_parts = ["python", script] + list(args)
    python_cmd = " ".join(shlex.quote(p) for p in python_parts)
    bash_cmd = f'eval "$(conda shell.bash hook)" && conda activate {CONDA_ENV} && {python_cmd}'
    return bash_cmd


def load_evaluations(filepath: str) -> list:
    """Load evaluation entries from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_costorm_output(base_dir: str, eval_id: str) -> str | None:
    """Find the most recent CoStorm output directory for a given evaluation."""
    eval_dir = os.path.join(base_dir, eval_id)

    if not os.path.exists(eval_dir):
        return None

    candidates = []
    for item in os.listdir(eval_dir):
        item_path = os.path.join(eval_dir, item)
        if os.path.isdir(item_path):
            report_file = os.path.join(item_path, "co_storm_report.txt")
            if os.path.exists(report_file):
                candidates.append((item_path, os.path.getmtime(item_path)))

    direct_report = os.path.join(eval_dir, "co_storm_report.txt")
    if os.path.exists(direct_report):
        return direct_report

    if not candidates:
        pattern = os.path.join(base_dir, "*", "co_storm_report.txt")
        all_reports = glob.glob(pattern)
        if all_reports:
            all_reports.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return all_reports[0]
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(candidates[0][0], "co_storm_report.txt")


def write_single_row_csv(csv_path: str, *, fieldnames: list[str], row: dict):
    """Write a single-row CSV file."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


CSV_FIELDNAMES = [
    "id",
    "prompt",
    "total_insights",
    "acled_count",
    "non_acled_count",
    "pct_acled",
]


def run_data_source_evaluator(
    prompt: str,
    generated_file: str,
    output_file: str,
) -> dict | None:
    """Run the data source evaluator script and return the results."""
    if not os.path.exists(generated_file):
        print(f"[Data Source] Generated file not found: {generated_file}")
        return None

    cmd = build_conda_shell_cmd(
        DATA_SOURCE_EVALUATOR_SCRIPT,
        "--prompt", prompt,
        "--generated_file", generated_file,
        "--output_file", output_file,
    )
    print(f"[Data Source] Running: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[Data Source] Failed with return code {result.returncode}")
        print(f"[Data Source] stderr: {result.stderr}")
        return None

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def _print_result_summary(label: str, r: dict):
    """Print a concise summary of a single evaluation result."""
    print(f"{label}:")
    print(f"  Total insights: {r.get('total_insights', 'N/A')}")
    print(f"  ACLED-sourced: {r.get('acled_count', 'N/A')}")
    print(f"  Non-ACLED: {r.get('non_acled_count', 'N/A')}")
    print(f"  % ACLED: {r.get('pct_acled', 0):.2%}")


def _result_to_csv_row(eval_id: str, prompt: str, r: dict) -> dict:
    """Convert a result dict to a CSV row dict."""
    return {
        "id": eval_id,
        "prompt": prompt,
        "total_insights": r.get("total_insights", ""),
        "acled_count": r.get("acled_count", ""),
        "non_acled_count": r.get("non_acled_count", ""),
        "pct_acled": r.get("pct_acled", ""),
    }


def _try_load_cached(path: str, label: str) -> dict | None:
    """Try to load a cached result JSON. Returns the parsed dict or None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[Data Source] {label}: pct_acled={data.get('pct_acled', 'N/A'):.2%} (cached)")
        return data
    except Exception as e:
        print(f"[Data Source] Failed to load cached result: {e}")
        return None


def _evaluate_single_source(
    eval_id: str,
    prompt: str,
    generated_file: str,
    eval_output: str,
    label: str,
    resume: bool,
) -> dict | None:
    """Evaluate a single generated file and return the result dict (or None)."""
    if not os.path.exists(generated_file):
        print(f"[Data Source] {label} output not found: {generated_file}")
        return None

    # Try cache
    if resume and os.path.exists(eval_output):
        cached = _try_load_cached(eval_output, label)
        if cached is not None:
            return cached

    print(f"\n[Data Source] Evaluating {label}...")
    result = run_data_source_evaluator(
        prompt=prompt,
        generated_file=generated_file,
        output_file=eval_output,
    )
    if result:
        print(f"[Data Source] {label} pct_acled: {result.get('pct_acled', 'N/A'):.2%}")
        write_single_row_csv(
            eval_output.replace(".json", ".csv"),
            fieldnames=CSV_FIELDNAMES,
            row=_result_to_csv_row(eval_id, prompt, result),
        )
    return result


def run_evaluation(
    evaluation: dict,
    skip_openai: bool = False,
    skip_costorm: bool = False,
    resume: bool = False,
):
    """Run data source evaluation for a single entry."""
    eval_id = evaluation["id"]
    prompt = evaluation["prompt"]

    print(f"\n{'='*60}")
    print(f"Running Data Source Evaluation ID: {eval_id}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")

    # Generated file paths
    openai_output_file = os.path.join(RUNS_DIR, f"{eval_id}.txt")
    openai_csv_output_file = os.path.join(RUNS_DIR, f"{eval_id}_csv.txt")
    costorm_report = find_latest_costorm_output(COSTORM_OUTPUT_DIR, eval_id)

    # Evaluation output paths
    os.makedirs(EVALUATIONS_OUTPUT_DIR, exist_ok=True)
    openai_eval_output = os.path.join(EVALUATIONS_OUTPUT_DIR, f"{eval_id}_openai_dr.json")
    openai_csv_eval_output = os.path.join(EVALUATIONS_OUTPUT_DIR, f"{eval_id}_openai_dr_csv.json")
    costorm_eval_output = os.path.join(EVALUATIONS_OUTPUT_DIR, f"{eval_id}_costorm.json")

    results = {
        "id": eval_id,
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "openai_dr": None,
        "openai_dr_csv": None,
        "costorm": None,
    }

    # Evaluate OpenAI DR
    if not skip_openai:
        results["openai_dr"] = _evaluate_single_source(
            eval_id, prompt,
            openai_output_file, openai_eval_output,
            "OpenAI DR", resume,
        )

    # Evaluate OpenAI DR CSV
    if not skip_openai:
        results["openai_dr_csv"] = _evaluate_single_source(
            eval_id, prompt,
            openai_csv_output_file, openai_csv_eval_output,
            "OpenAI DR CSV", resume,
        )

    # Evaluate CoStorm
    if not skip_costorm and costorm_report:
        results["costorm"] = _evaluate_single_source(
            eval_id, prompt,
            costorm_report, costorm_eval_output,
            "CoStorm", resume,
        )
    elif not skip_costorm:
        print(f"[Data Source] CoStorm output not found in: {COSTORM_OUTPUT_DIR}/{eval_id}")

    # Save combined results
    combined_output = os.path.join(EVALUATIONS_OUTPUT_DIR, f"{eval_id}.json")
    with open(combined_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Data Source] Combined results saved to: {combined_output}")

    # Print summary
    print(f"\n{'='*40}")
    print(f"Data Source Summary for ID: {eval_id}")
    print(f"{'='*40}")

    if results["openai_dr"]:
        _print_result_summary("OpenAI DR", results["openai_dr"])
    if results["openai_dr_csv"]:
        _print_result_summary("OpenAI DR (CSV)", results["openai_dr_csv"])
    if results["costorm"]:
        _print_result_summary("CoStorm", results["costorm"])

    return results


def _print_latex_table(per_eval: dict, summary: dict, sources: dict):
    """Print a LaTeX table summarizing data source classification results."""
    labels = ["OpenAI DR", "OpenAI DR (CSV)", "CoStorm"]
    eval_ids = sorted(per_eval.keys(), key=lambda x: int(x) if x.isdigit() else x)

    print(f"\n{'='*60}")
    print("LaTeX Table")
    print(f"{'='*60}\n")

    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{ACLED Data Source Classification}")
    print(r"\label{tab:data-source}")

    # Per-evaluation detailed table
    col_spec = "l" + "rrr" * len(labels)
    print(r"\begin{tabular}{" + col_spec + "}")
    print(r"\toprule")

    # Header row 1: source names spanning 3 columns each
    header1 = "ID"
    for label in labels:
        header1 += f" & \\multicolumn{{3}}{{c}}{{{label}}}"
    header1 += r" \\"
    print(header1)

    # Header row 2: sub-columns
    cmidrule_parts = []
    for i, _ in enumerate(labels):
        start = 2 + i * 3
        cmidrule_parts.append(f"\\cmidrule(lr){{{start}-{start + 2}}}")
    print(" ".join(cmidrule_parts))

    header2 = ""
    for _ in labels:
        header2 += " & ACLED & Total & \\% ACLED"
    header2 += r" \\"
    print(header2)
    print(r"\midrule")

    # Data rows
    for eval_id in eval_ids:
        row = eval_id
        for label in labels:
            r = per_eval[eval_id].get(label)
            if r:
                acled = r.get("acled_count", 0)
                total = r.get("total_insights", 0)
                pct = r.get("pct_acled", 0)
                row += f" & {acled} & {total} & {pct * 100:.1f}\\%"
            else:
                row += " & -- & -- & --"
        row += r" \\"
        print(row)

    # Average row
    print(r"\midrule")
    avg_row = r"\textbf{Avg}"
    for label in labels:
        s = summary.get(label)
        if s:
            avg_row += f" & {s['total_acled']} & {s['total_insights']} & {s['avg_pct'] * 100:.1f}\\%"
        else:
            avg_row += " & -- & -- & --"
    avg_row += r" \\"
    print(avg_row)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def compute_averages(evaluations_dir: str):
    """Compute average data source scores across all evaluations."""
    sources = {
        "OpenAI DR": {"pct_acled": [], "acled_count": [], "non_acled_count": [], "total": []},
        "OpenAI DR (CSV)": {"pct_acled": [], "acled_count": [], "non_acled_count": [], "total": []},
        "CoStorm": {"pct_acled": [], "acled_count": [], "non_acled_count": [], "total": []},
    }
    key_map = {
        "OpenAI DR": "openai_dr",
        "OpenAI DR (CSV)": "openai_dr_csv",
        "CoStorm": "costorm",
    }
    # Per-evaluation rows: {eval_id: {source_label: result_dict}}
    per_eval = {}

    if not os.path.exists(evaluations_dir):
        print(f"[Warning] Evaluations directory not found: {evaluations_dir}")
        return

    for filename in sorted(os.listdir(evaluations_dir)):
        if not filename.endswith(".json"):
            continue
        if "_openai_dr" in filename or "_costorm" in filename:
            continue

        filepath = os.path.join(evaluations_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            eval_id = data.get("id", filename.replace(".json", ""))
            per_eval[eval_id] = {}

            for label, key in key_map.items():
                r = data.get(key)
                if r and "pct_acled" in r:
                    sources[label]["pct_acled"].append(r["pct_acled"])
                    sources[label]["acled_count"].append(r.get("acled_count", 0))
                    sources[label]["non_acled_count"].append(r.get("non_acled_count", 0))
                    sources[label]["total"].append(r.get("total_insights", 0))
                    per_eval[eval_id][label] = r
        except Exception as e:
            print(f"[Warning] Failed to read {filepath}: {e}")
            continue

    print(f"\n{'='*60}")
    print("Average Data Source Classification Scores")
    print(f"{'='*60}")

    summary = {}
    for label, vals in sources.items():
        if vals["pct_acled"]:
            n = len(vals["pct_acled"])
            avg_pct = sum(vals["pct_acled"]) / n
            total_acled = sum(vals["acled_count"])
            total_insights = sum(vals["total"])
            print(f"{label} (n={n}):")
            print(f"  Avg % ACLED: {avg_pct:.2%}")
            print(f"  Total ACLED insights: {total_acled}")
            print(f"  Total insights: {total_insights}")
            summary[label] = {
                "n": n, "avg_pct": avg_pct,
                "total_acled": total_acled, "total_insights": total_insights,
            }

    print(f"{'='*60}")

    # LaTeX table
    _print_latex_table(per_eval, summary, sources)


def main():
    parser = argparse.ArgumentParser(description="Run data source evaluations")
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Specific evaluation IDs to run (default: all)",
    )
    parser.add_argument(
        "--skip-openai",
        action="store_true",
        help="Skip evaluating OpenAI Deep Research outputs",
    )
    parser.add_argument(
        "--skip-costorm",
        action="store_true",
        help="Skip evaluating CoStorm outputs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip evaluations that already have output files",
    )
    parser.add_argument(
        "--compute-averages-only",
        action="store_true",
        help="Only compute averages from existing evaluation results",
    )
    args = parser.parse_args()

    if args.compute_averages_only:
        compute_averages(EVALUATIONS_OUTPUT_DIR)
        return

    # Load evaluations
    evaluations = load_evaluations(EVALUATIONS_FILE)
    print(f"Loaded {len(evaluations)} evaluations from {EVALUATIONS_FILE}")

    # Filter by IDs if specified
    if args.ids:
        evaluations = [e for e in evaluations if e["id"] in args.ids]
        print(f"Filtered to {len(evaluations)} evaluations: {args.ids}")

    if not evaluations:
        print("No evaluations to run")
        return

    # Run each evaluation
    for evaluation in evaluations:
        try:
            run_evaluation(
                evaluation,
                skip_openai=args.skip_openai,
                skip_costorm=args.skip_costorm,
                resume=args.resume,
            )
        except KeyboardInterrupt:
            print("\n[Interrupted] Stopping evaluations...")
            break
        except Exception as e:
            print(f"[Error] Evaluation {evaluation['id']} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute averages at the end
    compute_averages(EVALUATIONS_OUTPUT_DIR)

    print("\n" + "="*60)
    print("All evaluations completed")
    print("="*60)


if __name__ == "__main__":
    main()
