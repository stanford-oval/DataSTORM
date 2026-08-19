"""Ablation runner for DataSTORM on the ACLED benchmark (paper section 4.2).

Calls `datatalk_basic_article_generation_costorm.py` directly as a subprocess
with the paper baseline config (depth=8, first_level=8, warm-start ON, full
editorial pipeline ON, Co-STORM Serper retriever with per-topic date restriction).

Each ablation flips exactly one CLI flag relative to baseline:
  baseline      — no extra flags
  no_qc         — --disable_followups
  no_inductive  — --no_summary_stats
  no_thesis     — --skip_thesis
  no_warm_start — --no_warm_start

Output for each (ID, ablation) lands under:
  {datastorm_run_dir}/{id}/{slug}__{ts}/   — tree.json, co_storm_report*, etc.

Scoring against paper metrics (criteria match, RACE, DB-use ratio, originality,
consistency) is done post-hoc via the existing eval pipelines in
breakdown_evaluator/ and query_consistency_eval/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOPICS_PATH = HERE / "topics.json"

# DataSTORM repo root (where datatalk_basic_article_generation_costorm.py lives).
# datatalk is now integrated in-process, so the orchestrator lives at the storm
# repo root (the parent of this acled_ablation/ dir) rather than a separate repo.
DATASTORM_REPO_ROOT = os.environ.get(
    "DATASTORM_REPO_ROOT", str(HERE.parent)
)
ORCHESTRATOR_SCRIPT = os.path.join(
    DATASTORM_REPO_ROOT, "datatalk_basic_article_generation_costorm.py"
)

ABLATION_CLI_FLAGS = {
    "baseline":      [],
    "no_qc":         ["--disable_followups"],
    "no_inductive":  ["--no_summary_stats"],
    "no_thesis":     ["--skip_thesis"],
    "no_warm_start": ["--no_warm_start"],
}


def load_topics():
    with open(TOPICS_PATH) as f:
        return json.load(f)


def run_one(
    *,
    instance_id,
    topic,
    tbs,
    ablation,
    run_dir,
    max_tree_depth,
    first_level_questions,
    each_level_population_control_num,
    max_global_insights,
    expansion_max_questions,
    generation_module_model,
    datatalk_engine,
    datastorm_main_model,
    skip_generated,
    subprocess_timeout,
):
    output_dir = os.path.join(run_dir, str(instance_id))
    os.makedirs(output_dir, exist_ok=True)

    if skip_generated:
        existing = list(Path(output_dir).glob("*/tree.json"))
        if existing:
            print(f"[id={instance_id} {ablation}] reusing tree at {existing[0]}")
            return

    serper_params = json.dumps({"tbs": tbs})
    cmd = [
        sys.executable, ORCHESTRATOR_SCRIPT,
        "--output_dir", output_dir,
        "--topic", topic,
        "--domain", "acled",
        "--serper_query_params", serper_params,
        "--first_level_questions", str(first_level_questions),
        "--max_tree_depth", str(max_tree_depth),
        "--each_level_population_control_num", str(each_level_population_control_num),
        "--max_global_insights", str(max_global_insights),
        "--expansion_max_questions", str(expansion_max_questions),
        "--generation_module_model", generation_module_model,
        "--datatalk_engine", datatalk_engine,
        "--datastorm_main_model", datastorm_main_model,
        # Paper used Azure CSV upload + graph generation. We disable both
        # locally: no azcopy installed, and graph gen requires a docker
        # sidecar we don't need for the ablation comparison.
        "--disable_upload_to_azure",
        "--disable_graphs",
    ] + ABLATION_CLI_FLAGS[ablation]

    prefix = f"[id={instance_id} {ablation}] "
    stderr_tail = deque(maxlen=200)

    def _stream(src, dst, *, buffer=None):
        for line in iter(src.readline, ""):
            dst.write(prefix + line)
            dst.flush()
            if buffer is not None:
                buffer.append(line)
        src.close()

    print(f"{prefix}launching: max_tree_depth={max_tree_depth} flags={ABLATION_CLI_FLAGS[ablation] or '(baseline)'}")
    sys.stdout.flush()
    t_start = time.time()
    # PYTHONUNBUFFERED=1: without this, the orchestrator's print() output sits in
    # block-buffered stdout (~4-8KB) when piped to our parent, so the live stream
    # only flushes once that buffer fills — making the tee'd log file appear
    # frozen for long stretches even though the run is making progress.
    # PYTHONWARNINGS: silence the noisy "Pydantic serializer warnings: Expected
    # `none` but got `LLMThoughtAction`/`SqlReporterResponse`/..." prints emitted
    # by pydantic.main on every .to_python() call in the datatalk_agent code.
    # The warnings indicate union-tag mismatches in those models but don't affect
    # output; suppressing only this message keeps unrelated UserWarnings visible.
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
        bufsize=1,
        env=child_env,
    )
    t_out = threading.Thread(target=_stream, args=(proc.stdout, sys.stdout), daemon=True)
    t_err = threading.Thread(target=_stream, args=(proc.stderr, sys.stderr),
                              kwargs={"buffer": stderr_tail}, daemon=True)
    t_out.start(); t_err.start()
    try:
        rc = proc.wait(timeout=subprocess_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise RuntimeError(
            f"ACLED subprocess timed out after {subprocess_timeout}s for id={instance_id} ablation={ablation}"
        )
    t_out.join(timeout=5); t_err.join(timeout=5)
    elapsed = time.time() - t_start
    if rc != 0:
        raise RuntimeError(
            f"ACLED subprocess failed (rc={rc}) for id={instance_id} ablation={ablation}.\n"
            f"--- stderr tail ---\n{''.join(stderr_tail)}"
        )
    print(f"{prefix}finished in {elapsed:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", required=True, choices=list(ABLATION_CLI_FLAGS.keys()))
    ap.add_argument("--ids", type=int, nargs="+", required=True,
                    help="ACLED instance IDs (1-20). See topics.json for the topic mapping.")
    ap.add_argument("--datastorm_run_dir", required=True,
                    help="Where DataSTORM dumps {id}/{slug}__{ts}/tree.json etc.")
    # Match the paper baseline config (extracted from run_metadata.json for the
    # 20 published ACLED runs). Most callers will want to leave these alone.
    ap.add_argument("--max_tree_depth", type=int, default=8)
    ap.add_argument("--first_level_questions", type=int, default=8)
    ap.add_argument("--each_level_population_control_num", type=int, default=3)
    ap.add_argument("--max_global_insights", type=int, default=30)
    ap.add_argument("--expansion_max_questions", type=int, default=5)
    ap.add_argument("--generation_module_model", default="gpt-5.1",
                    help="Model used by the 5-stage editorial pipeline (paper used gpt-5.1).")
    ap.add_argument("--datatalk_engine", default="gpt-5")
    ap.add_argument("--datastorm_main_model", default="gpt-5")
    ap.add_argument("--no_skip_generated", action="store_true",
                    help="Force regenerate even if a tree already exists under output_dir.")
    ap.add_argument("--subprocess_timeout", type=int, default=14400,
                    help="Per-instance hard timeout in seconds (default 4h).")
    args = ap.parse_args()

    topics = load_topics()
    print(f"[config] ablation={args.ablation} ids={args.ids} run_dir={args.datastorm_run_dir}")

    for i in args.ids:
        spec = topics.get(str(i))
        if spec is None:
            raise SystemExit(f"No topic mapping for id={i}. See {TOPICS_PATH}.")
        try:
            run_one(
                instance_id=i,
                topic=spec["topic"],
                tbs=spec["tbs"],
                ablation=args.ablation,
                run_dir=args.datastorm_run_dir,
                max_tree_depth=args.max_tree_depth,
                first_level_questions=args.first_level_questions,
                each_level_population_control_num=args.each_level_population_control_num,
                max_global_insights=args.max_global_insights,
                expansion_max_questions=args.expansion_max_questions,
                generation_module_model=args.generation_module_model,
                datatalk_engine=args.datatalk_engine,
                datastorm_main_model=args.datastorm_main_model,
                skip_generated=not args.no_skip_generated,
                subprocess_timeout=args.subprocess_timeout,
            )
        except Exception as e:
            print(f"[id={i} {args.ablation}] FAILED: {e}", file=sys.stderr)
            continue


if __name__ == "__main__":
    main()
