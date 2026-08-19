#!/usr/bin/env python3
"""
Insight Recall Evaluation Script

This script runs insight recall evaluations comparing generated articles against references.
It breaks down each article into atomic insights and measures what fraction of reference
insights are captured in the generated article.

Usage:
    python run_insight_recall_evaluations.py                    # Run all evaluations
    python run_insight_recall_evaluations.py --ids 1 2         # Run specific evaluation IDs
    python run_insight_recall_evaluations.py --skip-costorm    # Only evaluate OpenAI outputs
    python run_insight_recall_evaluations.py --skip-openai     # Only evaluate CoStorm outputs
    python run_insight_recall_evaluations.py --resume          # Skip already completed evaluations
"""

import json
import os
import pathlib
import csv
import argparse
import glob
from datetime import datetime

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STORM_ROOT = os.path.dirname(SCRIPT_DIR)
OPENAI_DR_BASELINE_DIR = os.path.join(STORM_ROOT, "openai_dr_baseline")
EVALUATIONS_FILE = os.path.join(OPENAI_DR_BASELINE_DIR, "acled_evaluations.json")
RUNS_DIR = os.path.join(OPENAI_DR_BASELINE_DIR, "runs")
EVALUATIONS_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "insight_recall_evaluations")
COSTORM_OUTPUT_DIR = os.environ.get(
    "DATASTORM_ACLED_RUNS",
    str(pathlib.Path(__file__).resolve().parents[2] / "results" / "acled" / "runs"),
)

INSIGHT_RECALL_EVALUATOR_SCRIPT = os.path.join(SCRIPT_DIR, "insight_recall_evaluator.py")

# Conda environment name
CONDA_ENV = "storm"


def build_conda_shell_cmd(script: str, *args: str) -> str:
    """Build a shell command string that runs a Python script in the storm conda environment."""
    import shlex
    python_parts = ["python", script] + list(args)
    python_cmd = " ".join(shlex.quote(p) for p in python_parts)
    bash_cmd = f'eval "$(conda shell.bash hook)" && conda activate {CONDA_ENV} && {python_cmd}'
    return bash_cmd


def load_evaluations(filepath: str) -> list:
    """Load evaluation entries from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_costorm_output(base_dir: str, eval_id: str) -> str | None:
    """
    Find the most recent CoStorm output directory for a given evaluation.
    """
    eval_dir = os.path.join(base_dir, eval_id)

    if not os.path.exists(eval_dir):
        return None

    # Find all subdirectories with co_storm_report.txt
    candidates = []
    for item in os.listdir(eval_dir):
        item_path = os.path.join(eval_dir, item)
        if os.path.isdir(item_path):
            report_file = os.path.join(item_path, "co_storm_report.txt")
            if os.path.exists(report_file):
                candidates.append((item_path, os.path.getmtime(item_path)))

    # Also check if eval_dir itself has co_storm_report.txt
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


def run_insight_recall_evaluator(
    prompt: str,
    reference_file: str,
    generated_file: str,
    output_file: str,
    confidence_threshold: float = 0.7,
) -> dict | None:
    """
    Run the insight recall evaluator script and return the results.
    """
    import subprocess
    import shlex

    if not os.path.exists(generated_file):
        print(f"[Insight Recall] Generated file not found: {generated_file}")
        return None

    if not os.path.exists(reference_file):
        print(f"[Insight Recall] Reference file not found: {reference_file}")
        return None

    cmd = build_conda_shell_cmd(
        INSIGHT_RECALL_EVALUATOR_SCRIPT,
        "--prompt", prompt,
        "--reference_file", reference_file,
        "--generated_file", generated_file,
        "--output_file", output_file,
        "--confidence_threshold", str(confidence_threshold),
    )
    print(f"[Insight Recall] Running: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[Insight Recall] Failed with return code {result.returncode}")
        print(f"[Insight Recall] stderr: {result.stderr}")
        return None

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def run_evaluation(
    evaluation: dict,
    skip_openai: bool = False,
    skip_costorm: bool = False,
    resume: bool = False,
    confidence_threshold: float = 0.7,
):
    """Run insight recall evaluation for a single entry."""
    eval_id = evaluation["id"]
    prompt = evaluation["prompt"]
    reference_file = evaluation["reference_file"]

    print(f"\n{'='*60}")
    print(f"Running Insight Recall Evaluation ID: {eval_id}")
    print(f"Prompt: {prompt}")
    print(f"Reference file: {reference_file}")
    print(f"{'='*60}\n")

    # Define file paths
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
        "reference_file": reference_file,
        "timestamp": datetime.now().isoformat(),
        "openai_dr": None,
        "openai_dr_csv": None,
        "costorm": None,
    }

    csv_fieldnames = [
        "id",
        "prompt",
        "recall",
        "matched_count",
        "total_reference_insights",
        "total_predicted_insights",
    ]

    # Evaluate OpenAI Deep Research output
    if not skip_openai and os.path.exists(openai_output_file):
        if resume and os.path.exists(openai_eval_output):
            print(f"[Insight Recall] Loading cached OpenAI DR evaluation: {openai_eval_output}")
            try:
                with open(openai_eval_output, "r", encoding="utf-8") as f:
                    results["openai_dr"] = json.load(f)
                print(f"[Insight Recall] OpenAI DR Recall: {results['openai_dr'].get('recall', 'N/A'):.4f} (cached)")
            except Exception as e:
                print(f"[Insight Recall] Failed to load cached result: {e}")
                results["openai_dr"] = None

        if results["openai_dr"] is None:
            print(f"\n[Insight Recall] Evaluating OpenAI Deep Research output...")
            result = run_insight_recall_evaluator(
                prompt=prompt,
                reference_file=reference_file,
                generated_file=openai_output_file,
                output_file=openai_eval_output,
                confidence_threshold=confidence_threshold,
            )
            if result:
                results["openai_dr"] = result
                print(f"[Insight Recall] OpenAI DR Recall: {result.get('recall', 'N/A'):.4f}")
                # Write CSV summary
                write_single_row_csv(
                    openai_eval_output.replace(".json", ".csv"),
                    fieldnames=csv_fieldnames,
                    row={
                        "id": eval_id,
                        "prompt": prompt,
                        "recall": result.get("recall", ""),
                        "matched_count": result.get("matched_count", ""),
                        "total_reference_insights": result.get("total_reference_insights", ""),
                        "total_predicted_insights": result.get("total_predicted_insights", ""),
                    },
                )
    elif not skip_openai:
        print(f"[Insight Recall] OpenAI DR output not found: {openai_output_file}")

    # Evaluate OpenAI Deep Research CSV output
    if not skip_openai and os.path.exists(openai_csv_output_file):
        if resume and os.path.exists(openai_csv_eval_output):
            print(f"[Insight Recall] Loading cached OpenAI DR CSV evaluation: {openai_csv_eval_output}")
            try:
                with open(openai_csv_eval_output, "r", encoding="utf-8") as f:
                    results["openai_dr_csv"] = json.load(f)
                print(f"[Insight Recall] OpenAI DR CSV Recall: {results['openai_dr_csv'].get('recall', 'N/A'):.4f} (cached)")
            except Exception as e:
                print(f"[Insight Recall] Failed to load cached result: {e}")
                results["openai_dr_csv"] = None

        if results["openai_dr_csv"] is None:
            print(f"\n[Insight Recall] Evaluating OpenAI Deep Research CSV output...")
            result = run_insight_recall_evaluator(
                prompt=prompt,
                reference_file=reference_file,
                generated_file=openai_csv_output_file,
                output_file=openai_csv_eval_output,
                confidence_threshold=confidence_threshold,
            )
            if result:
                results["openai_dr_csv"] = result
                print(f"[Insight Recall] OpenAI DR CSV Recall: {result.get('recall', 'N/A'):.4f}")
                write_single_row_csv(
                    openai_csv_eval_output.replace(".json", ".csv"),
                    fieldnames=csv_fieldnames,
                    row={
                        "id": eval_id,
                        "prompt": prompt,
                        "recall": result.get("recall", ""),
                        "matched_count": result.get("matched_count", ""),
                        "total_reference_insights": result.get("total_reference_insights", ""),
                        "total_predicted_insights": result.get("total_predicted_insights", ""),
                    },
                )

    # Evaluate CoStorm output
    if not skip_costorm and costorm_report and os.path.exists(costorm_report):
        if resume and os.path.exists(costorm_eval_output):
            print(f"[Insight Recall] Loading cached CoStorm evaluation: {costorm_eval_output}")
            try:
                with open(costorm_eval_output, "r", encoding="utf-8") as f:
                    results["costorm"] = json.load(f)
                print(f"[Insight Recall] CoStorm Recall: {results['costorm'].get('recall', 'N/A'):.4f} (cached)")
            except Exception as e:
                print(f"[Insight Recall] Failed to load cached result: {e}")
                results["costorm"] = None

        if results["costorm"] is None:
            print(f"\n[Insight Recall] Evaluating CoStorm output: {costorm_report}")
            result = run_insight_recall_evaluator(
                prompt=prompt,
                reference_file=reference_file,
                generated_file=costorm_report,
                output_file=costorm_eval_output,
                confidence_threshold=confidence_threshold,
            )
            if result:
                results["costorm"] = result
                print(f"[Insight Recall] CoStorm Recall: {result.get('recall', 'N/A'):.4f}")
                write_single_row_csv(
                    costorm_eval_output.replace(".json", ".csv"),
                    fieldnames=csv_fieldnames,
                    row={
                        "id": eval_id,
                        "prompt": prompt,
                        "recall": result.get("recall", ""),
                        "matched_count": result.get("matched_count", ""),
                        "total_reference_insights": result.get("total_reference_insights", ""),
                        "total_predicted_insights": result.get("total_predicted_insights", ""),
                    },
                )
    elif not skip_costorm:
        print(f"[Insight Recall] CoStorm output not found in: {COSTORM_OUTPUT_DIR}/{eval_id}")

    # Save combined results
    combined_output = os.path.join(EVALUATIONS_OUTPUT_DIR, f"{eval_id}.json")
    with open(combined_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Insight Recall] Combined results saved to: {combined_output}")

    # Print summary
    print(f"\n{'='*40}")
    print(f"Insight Recall Summary for ID: {eval_id}")
    print(f"{'='*40}")

    if results["openai_dr"]:
        r = results["openai_dr"]
        print(f"OpenAI DR:")
        print(f"  Recall: {r.get('recall', 'N/A'):.4f} ({r.get('matched_count', 0)}/{r.get('total_reference_insights', 0)})")

    if results["openai_dr_csv"]:
        r = results["openai_dr_csv"]
        print(f"OpenAI DR (CSV):")
        print(f"  Recall: {r.get('recall', 'N/A'):.4f} ({r.get('matched_count', 0)}/{r.get('total_reference_insights', 0)})")

    if results["costorm"]:
        r = results["costorm"]
        print(f"CoStorm:")
        print(f"  Recall: {r.get('recall', 'N/A'):.4f} ({r.get('matched_count', 0)}/{r.get('total_reference_insights', 0)})")

    return results


def compute_averages(evaluations_dir: str):
    """Compute average recall scores across all evaluations."""
    openai_recalls = []
    openai_csv_recalls = []
    costorm_recalls = []

    for filename in os.listdir(evaluations_dir):
        if not filename.endswith(".json"):
            continue
        if "_openai_dr" in filename or "_costorm" in filename:
            continue  # Skip individual results, only process combined

        filepath = os.path.join(evaluations_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("openai_dr") and "recall" in data["openai_dr"]:
                openai_recalls.append(data["openai_dr"]["recall"])
            if data.get("openai_dr_csv") and "recall" in data["openai_dr_csv"]:
                openai_csv_recalls.append(data["openai_dr_csv"]["recall"])
            if data.get("costorm") and "recall" in data["costorm"]:
                costorm_recalls.append(data["costorm"]["recall"])
        except Exception as e:
            print(f"[Warning] Failed to read {filepath}: {e}")
            continue

    print(f"\n{'='*60}")
    print("Average Insight Recall Scores")
    print(f"{'='*60}")

    if openai_recalls:
        avg = sum(openai_recalls) / len(openai_recalls)
        print(f"OpenAI DR: {avg:.4f} (n={len(openai_recalls)})")

    if openai_csv_recalls:
        avg = sum(openai_csv_recalls) / len(openai_csv_recalls)
        print(f"OpenAI DR (CSV): {avg:.4f} (n={len(openai_csv_recalls)})")

    if costorm_recalls:
        avg = sum(costorm_recalls) / len(costorm_recalls)
        print(f"CoStorm: {avg:.4f} (n={len(costorm_recalls)})")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Run insight recall evaluations")
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
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for insight matching (default: 0.7)",
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
                confidence_threshold=args.confidence_threshold,
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
