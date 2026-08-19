"""Ablation runner for DataSTORM on InsightBench.

Calls `datatalk_basic_article_generation_costorm.py` directly as a subprocess
(no FastAPI server required). Each instance gets a fresh Python process,
which avoids server-side state accumulation and makes parallel runs trivial.

  1. --ablation {baseline,no_qc,no_thesis,no_inductive} maps to the orchestrator's
     CLI flags (--disable_followups / --no_summary_stats / --skip_thesis).
  2. --insight_source {follow_up,final_selected,both} controls how predicted_insights
     are extracted from the tree. no_qc requires final_selected (or both) since
     no follow-up nodes are produced when enable_followups=False.
  3. --ids restricts the benchmark to a specified list of dataset IDs (1-indexed).

The script then runs insight + summary g_eval scoring (matching the paper's
Table 1 metrics), including the Prompt-9 LLM-summarized variant.
"""

import os, argparse, json, time, sys
import pandas as pd
from collections import deque

# Locate the InsightBench repo (Sahu et al., ICLR 2025). We need its
# `insightbench` package + `data/notebooks/flag-*.json` benchmark files.
# Default points at the local clone; override with INSIGHT_BENCH_ROOT env var.
INSIGHT_BENCH_ROOT = os.environ.get("INSIGHT_BENCH_ROOT", "")
if not os.path.isdir(os.path.join(INSIGHT_BENCH_ROOT, "insightbench")):
    raise SystemExit(
        f"insightbench package not found at {INSIGHT_BENCH_ROOT}/insightbench. "
        "Set INSIGHT_BENCH_ROOT to your InsightBench clone."
    )
sys.path.insert(0, INSIGHT_BENCH_ROOT)

from insightbench import benchmarks  # noqa: F401
from insightbench.utils.exp_utils import save_json
from openai import OpenAI


# ------------------------- LLM summary helpers -------------------------

_SUMMARIZE_INDIV_SYSTEM = (
    "Your task is to summarize the findings of an insight into a succinct form. "
    "No need to include how many rows of data are used in the insight."
)
_SUMMARIZE_INDIV_EXAMPLE_INPUT = (
    "The database contains **56 rows of data**, including the 51 omitted rows. It tracks the number "
    "of incidents for different categories (e.g., Hardware, Network, Software, etc.) by month based "
    "on the `opened_at` date. \n\nFrom the visible data, we observe that **Hardware** consistently "
    "reports the highest number of incidents across months, with a peak of 28 incidents in January "
    "2023 and 19 incidents in January 2024."
)
_SUMMARIZE_INDIV_EXAMPLE_OUTPUT = "The Hardware incidents is significantly higher in volume than others"

_SUMMARIZE_LLM_PROMPT = (
    "You are given a list of data insights derived from a dataset analysis. "
    "Write a concise, coherent paragraph (3-5 sentences) that summarizes the key findings. "
    "Focus on the most important patterns and avoid repetition.\n\n"
    "Insights:\n{insights}"
)


def _openai_client():
    return OpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    )


def summarize_each_insight(pred_insights, model_name="gpt-4o"):
    if not pred_insights:
        return []
    client = _openai_client()
    out = []
    for ins in pred_insights:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _SUMMARIZE_INDIV_SYSTEM},
                {"role": "user", "content": _SUMMARIZE_INDIV_EXAMPLE_INPUT},
                {"role": "assistant", "content": _SUMMARIZE_INDIV_EXAMPLE_OUTPUT},
                {"role": "user", "content": ins},
            ],
            max_tokens=4000, temperature=0, top_p=0.9,
        )
        out.append(resp.choices[0].message.content)
    return out


def llm_predicted_summary(pred_insights, model_name="gpt-4o"):
    if not pred_insights:
        return ""
    numbered = "\n".join(f"{i+1}. {x}" for i, x in enumerate(pred_insights))
    prompt = _SUMMARIZE_LLM_PROMPT.format(insights=numbered)
    client = _openai_client()
    temperature = 1 if "gpt-5" in model_name.lower() else 0
    resp = client.chat.completions.create(
        model=model_name, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ------------------------- Tree → insights extraction -------------------------

def extract_insights(tree, source):
    if source == "follow_up":
        keep = lambda n: bool(n.get("is_follow_up"))
    elif source == "final_selected":
        keep = lambda n: bool(n.get("is_final_selected"))
    elif source == "both":
        keep = lambda n: bool(n.get("is_follow_up")) or bool(n.get("is_final_selected"))
    else:
        raise ValueError(f"Unknown insight_source {source!r}")
    out = []
    q = deque(tree.get("children", []) or [])
    while q:
        node = q.popleft()
        if keep(node):
            summary = (node.get("dlg_turn") or {}).get("summary")
            if summary:
                out.append(summary)
        q.extend(node.get("children", []) or [])
    return out


# ------------------------- DataSTORM call -------------------------

# Per-ablation flags that map to CLI args of datatalk_basic_article_generation_costorm.py.
# Each entry is a list of CLI tokens that get appended to the base command.
ABLATION_CLI_FLAGS = {
    "baseline":     [],                                       # all components on
    "no_qc":        ["--disable_followups"],                  # Query Consistency module off
    "no_inductive": ["--no_summary_stats"],                   # bottom-up summary stats off
    "no_thesis":    ["--skip_thesis"],                        # thesis generation/refinement off
}

# Default location of the DataSTORM orchestrator script. Override with
# DATASTORM_REPO_ROOT env var to point at a different clone.
DATASTORM_REPO_ROOT = os.environ.get(
    "DATASTORM_REPO_ROOT",
    str(pathlib.Path(__file__).resolve().parents[2]),
)
ORCHESTRATOR_SCRIPT = os.path.join(
    DATASTORM_REPO_ROOT, "datatalk_basic_article_generation_costorm.py"
)


def run_one(
    dataset_dict, i, run_dir, *,
    ablation, insight_source,
    max_tree_depth=6,
    first_level_questions=2,
    each_level_population_control_num=3,
    max_global_insights=30,
    expansion_max_questions=5,
    skip_generated=True,
    subprocess_timeout=14400,
):
    """Generate a DataSTORM tree for one InsightBench instance via subprocess.

    Calls datatalk_basic_article_generation_costorm.py directly (CLI mode), not
    through the FastAPI server. Reuses an existing tree.json under output_dir
    if skip_generated is True.
    """
    import subprocess, glob as g

    output_dir = os.path.join(run_dir, str(i))
    question = dataset_dict["metadata"]["goal"]

    def find_tree():
        m = g.glob(os.path.join(output_dir, "*", "tree.json"))
        return m[0] if m else None

    existing = find_tree()
    if existing and skip_generated:
        print(f"  reusing tree at {existing}")
    else:
        cmd = [
            sys.executable, ORCHESTRATOR_SCRIPT,
            "--output_dir", output_dir,
            "--topic", question,
            "--domain", f"insight_bench/insight_bench_{i}",
            "--first_level_questions", str(first_level_questions),
            "--max_tree_depth", str(max_tree_depth),
            "--each_level_population_control_num", str(each_level_population_control_num),
            "--max_global_insights", str(max_global_insights),
            "--expansion_max_questions", str(expansion_max_questions),
            "--no_warm_start",
            "--skip_final_article",
            "--disable_graphs",
            "--disable_upload_to_azure",
            "--datatalk_engine", "gpt-5",
            "--datastorm_main_model", "gpt-5",
        ] + ABLATION_CLI_FLAGS[ablation]

        # Prefix subprocess output with [id=N ablation=X] so parallel sweep logs
        # remain readable. Tee stderr through a buffer too so we can include a
        # tail on failure without losing live visibility.
        prefix = f"[id={i} {ablation}] "
        stderr_tail = deque(maxlen=200)

        def _stream(src, dst, *, buffer=None):
            for line in iter(src.readline, ""):
                dst.write(prefix + line)
                dst.flush()
                if buffer is not None:
                    buffer.append(line)
            src.close()

        print(f"{prefix}launching: {' '.join(cmd[:4])} ... ({len(cmd)} args)")
        sys.stdout.flush()
        t_proc = time.time()
        # PYTHONUNBUFFERED=1: without this, the orchestrator's print() output sits
        # in block-buffered stdout when piped, so the tee'd log file appears frozen
        # for long stretches even though the run is making progress.
        # PYTHONWARNINGS: silence the noisy "Pydantic serializer warnings: Expected
        # `none` but got `LLMThoughtAction`/`SqlReporterResponse`/..." prints from
        # pydantic.main; union-tag mismatch in the datatalk_agent models that
        # doesn't affect output.
        child_env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONWARNINGS": "ignore:Pydantic serializer warnings:UserWarning",
        }
        proc = subprocess.Popen(
            cmd,
            cwd=DATASTORM_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered on our side
            env=child_env,
        )
        import threading
        t_out = threading.Thread(target=_stream, args=(proc.stdout, sys.stdout), daemon=True)
        t_err = threading.Thread(target=_stream, args=(proc.stderr, sys.stderr), kwargs={"buffer": stderr_tail}, daemon=True)
        t_out.start(); t_err.start()
        try:
            rc = proc.wait(timeout=subprocess_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError(
                f"DataSTORM subprocess timed out after {subprocess_timeout}s for id={i} ablation={ablation}"
            )
        t_out.join(timeout=5); t_err.join(timeout=5)
        elapsed_proc = time.time() - t_proc
        if rc != 0:
            raise RuntimeError(
                f"DataSTORM subprocess failed (rc={rc}) for id={i} ablation={ablation}.\n"
                f"--- stderr tail ---\n{''.join(stderr_tail)}"
            )
        print(f"  subprocess finished in {elapsed_proc:.0f}s")
        existing = find_tree()

    if not existing:
        raise FileNotFoundError(f"Tree file not found under {output_dir}")

    with open(existing) as f:
        tree = json.load(f)
    raw_insights = extract_insights(tree, insight_source)
    summarized = summarize_each_insight(raw_insights)
    pred_summary_raw = "\n".join(summarized)
    return summarized, pred_summary_raw, tree


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", required=True, choices=list(ABLATION_CLI_FLAGS.keys()))
    ap.add_argument("--insight_source", default=None,
                    choices=["follow_up", "final_selected", "both"],
                    help="Default: follow_up for baseline/no_thesis/no_inductive; final_selected for no_qc.")
    ap.add_argument("--ids", type=int, nargs="+", required=True,
                    help="1-indexed dataset IDs to run.")
    ap.add_argument("--max_tree_depth", type=int, default=6,
                    help="Max tree-expansion depth passed to the orchestrator (default: 6).")
    ap.add_argument("--datadir", default=os.path.join(INSIGHT_BENCH_ROOT, "data/notebooks"))
    ap.add_argument("--eval_model_name", default="gpt-4o")
    ap.add_argument("--summarizer_model", default="gpt-4o",
                    help="Model used to write llm_predicted_summary (default gpt-4o, matches paper).")
    ap.add_argument("--benchmark_type", default="full")
    ap.add_argument("--no_skip_generated", action="store_true",
                    help="Force re-running DataSTORM even if tree.json exists.")
    ap.add_argument("--no_skip_scored", action="store_true",
                    help="Force re-scoring even if predictions/pred_gt_{i}.json with valid scores already exists.")
    ap.add_argument("--datastorm_run_dir", required=True,
                    help="Where DataSTORM should drop {i}/{slug}__{ts}/tree.json files.")
    ap.add_argument("--savedir", required=True,
                    help="Where to write predictions/pred_gt_{i}.json and score_list.json.")
    args = ap.parse_args()

    # Default insight_source per ablation
    if args.insight_source is None:
        args.insight_source = "final_selected" if args.ablation == "no_qc" else "follow_up"
    print(f"[config] ablation={args.ablation} insight_source={args.insight_source}")
    print(f"[config] run_dir={args.datastorm_run_dir}")
    print(f"[config] savedir={args.savedir}")
    print(f"[config] ids={args.ids}")

    os.makedirs(os.path.join(args.savedir, "predictions"), exist_ok=True)
    exp_dict = {
        "ablation": args.ablation,
        "insight_source": args.insight_source,
        "ids": args.ids,
        "model_name": "gpt-5",
        "eval_model_name": args.eval_model_name,
        "summarizer_model": args.summarizer_model,
        "benchmark_type": args.benchmark_type,
    }
    save_json(os.path.join(args.savedir, "exp_dict.json"), exp_dict)

    # Resolve dataset paths
    dataset_list = benchmarks.get_benchmark(args.benchmark_type, datadir=args.datadir)
    if len(dataset_list) < max(args.ids):
        raise SystemExit(f"benchmark has {len(dataset_list)} datasets but --ids includes {max(args.ids)}")

    score_list = []
    t0 = time.time()
    _REQUIRED_SCORE_KEYS = (
        "score_insights", "score_summary", "score_llm_predicted_summary",
        "predicted_insights",
    )
    for i in args.ids:
        # Reuse cached scores when predictions/pred_gt_{i}.json already has them.
        # Skipping here saves the gpt-4o judge cost (the slow part of scoring) for
        # IDs that were scored in a prior run of this same savedir. Override with
        # --no_skip_scored to force re-judging (e.g. after changing eval_model_name).
        pred_path = os.path.join(args.savedir, "predictions", f"pred_gt_{i}.json")
        if not args.no_skip_scored and os.path.exists(pred_path):
            try:
                with open(pred_path) as f:
                    cached = json.load(f)
                if all(k in cached for k in _REQUIRED_SCORE_KEYS):
                    score_list.append({
                        "id": i,
                        "score_insights": cached["score_insights"],
                        "score_summary": cached["score_summary"],
                        "score_llm_predicted_summary": cached["score_llm_predicted_summary"],
                        "elapsed_sec": 0.0,
                        "n_pred_insights": cached.get("n_pred_insights", len(cached["predicted_insights"])),
                    })
                    save_json(os.path.join(args.savedir, "score_list.json"), score_list)
                    print(f"[{i}] reusing cached scores from {pred_path}")
                    continue
            except Exception as e:
                print(f"[{i}] cached prediction unreadable, will re-score: {e}")

        dataset_json_path = dataset_list[i - 1]
        dataset_dict = benchmarks.load_dataset_dict(dataset_json_path=dataset_json_path)
        t_inst = time.time()
        try:
            pred_insights, pred_summary_raw, tree = run_one(
                dataset_dict, i, args.datastorm_run_dir,
                ablation=args.ablation, insight_source=args.insight_source,
                max_tree_depth=args.max_tree_depth,
                skip_generated=not args.no_skip_generated,
            )
        except Exception as e:
            print(f"[{i}] FAILED to run: {e}")
            continue

        # Score insights
        score_insights, score_insights_dict = benchmarks.evaluate_insights(
            pred_insights=pred_insights,
            gt_insights=dataset_dict["insights"],
            score_name="g_eval",
            model_name=args.eval_model_name,
            return_score_dict=True,
        )
        # Score raw concatenation summary (kept for parity with original)
        score_summary_raw = benchmarks.evaluate_summary(
            pred=pred_summary_raw, gt=dataset_dict["summary"],
            score_name="g_eval", model_name=args.eval_model_name,
        )
        # Score LLM-generated summary (the paper's reported summary metric)
        llm_summary = llm_predicted_summary(pred_insights, model_name=args.summarizer_model)
        score_llm_summary = benchmarks.evaluate_summary(
            pred=llm_summary, gt=dataset_dict["summary"],
            score_name="g_eval", model_name=args.eval_model_name,
        )

        prediction = {
            "dataset_id": i,
            "metadata": dataset_dict["metadata"],
            "dataset_path": dataset_json_path,
            "predicted_insights": pred_insights,
            "predicted_summary": pred_summary_raw,
            "ground_truth_insights": dataset_dict["insights"],
            "ground_truth_summary": dataset_dict["summary"],
            "score_insights": score_insights,
            "score_summary": score_summary_raw,
            "score_insights_dict": score_insights_dict,
            "llm_predicted_summary": llm_summary,
            "score_llm_predicted_summary": score_llm_summary,
            "ablation": args.ablation,
            "insight_source": args.insight_source,
            "n_pred_insights": len(pred_insights),
            "tree_path": next(iter(__import__("glob").glob(os.path.join(args.datastorm_run_dir, str(i), "*", "tree.json"))), None),
        }
        save_json(os.path.join(args.savedir, "predictions", f"pred_gt_{i}.json"), prediction)

        elapsed = time.time() - t_inst
        score_list.append({
            "id": i,
            "score_insights": score_insights,
            "score_summary": score_summary_raw,
            "score_llm_predicted_summary": score_llm_summary,
            "elapsed_sec": elapsed,
            "n_pred_insights": len(pred_insights),
        })
        save_json(os.path.join(args.savedir, "score_list.json"), score_list)
        df = pd.DataFrame(score_list)
        running_means = (df[["score_insights", "score_summary", "score_llm_predicted_summary"]].mean()
                         .to_dict())
        print(f"[{i}] insights={score_insights:.3f} summary_raw={score_summary_raw:.3f} "
              f"summary_llm={score_llm_summary:.3f} "
              f"n_insights={len(pred_insights)} elapsed={elapsed:.0f}s")
        print(f"    running means: " + ", ".join(f"{k}={v:.4f}" for k, v in running_means.items()))

    print(f"\nDone. Total wall-clock: {time.time() - t0:.0f}s")
    print(f"Final means across {len(score_list)} instances:")
    if score_list:
        df = pd.DataFrame(score_list)
        for k in ("score_insights", "score_summary", "score_llm_predicted_summary"):
            print(f"  {k}: {df[k].mean():.4f}")


if __name__ == "__main__":
    main()
