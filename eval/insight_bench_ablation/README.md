# InsightBench ablation runner

Driver script for component-level ablations of DataSTORM on the InsightBench
benchmark (Sahu et al., ICLR 2025). Reproduces the paper's evaluation
pipeline (insight-level g_eval + Prompt-9 LLM-summarized summary g_eval).

Calls the DataSTORM orchestrator (`datatalk_basic_article_generation_costorm.py`)
directly as a subprocess — no FastAPI server needed. Each instance runs in a
fresh Python process, so parallel sweeps just launch multiple subprocesses.

## Setup

**1. InsightBench upstream repo**: needs `insightbench` package, `data/notebooks/flag-*.json`,
and per-instance `flag-{i}` tables loaded into a Postgres database named `insight_bench`.

```bash
export INSIGHT_BENCH_ROOT=/path/to/insight_bench  # required
```

**2. DataSTORM repo**:

```bash
export DATASTORM_REPO_ROOT=/path/to/datastorm  # default: the repo root
```

**3. Credentials**:

```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=...   # any OpenAI-compatible endpoint
```

The script uses `sys.executable` to invoke the orchestrator, so the Python
environment must have both the DataSTORM dependencies and the InsightBench
package importable.

## Ablations

| `--ablation` | Orchestrator CLI flag added | Default `--insight_source` |
|--------------|-----------------------------|----------------------------|
| `baseline` | (none — all components on) | `follow_up` |
| `no_qc` | `--disable_followups` | `final_selected` (no follow-ups exist) |
| `no_inductive` | `--no_summary_stats` | `follow_up` |
| `no_thesis` | `--skip_thesis` | `follow_up` |

## Example

```bash
# Baseline on 5 IDs
python insight_bench_ablation/run_datastorm_ablation.py \
    --ablation baseline \
    --ids 4 5 12 14 15 \
    --datastorm_run_dir /path/to/runs/baseline \
    --savedir /path/to/results/baseline

# no_inductive — tests whether bottom-up summary stats help
python insight_bench_ablation/run_datastorm_ablation.py \
    --ablation no_inductive \
    --ids 4 5 12 14 15 \
    --datastorm_run_dir /path/to/runs/no_inductive \
    --savedir /path/to/results/no_inductive
```

To parallelize, just launch multiple invocations in parallel (different
`--datastorm_run_dir` + `--savedir` for different ablations).

## Output

Per ID:
- DataSTORM tree dump at `{datastorm_run_dir}/{id}/{slug}__{ts}/tree.json`
- One prediction file at `{savedir}/predictions/pred_gt_{id}.json`, with
  `predicted_insights`, `score_insights`, `score_insights_dict`,
  `predicted_summary`, `score_summary`, `llm_predicted_summary`,
  `score_llm_predicted_summary`
- An aggregated `{savedir}/score_list.json` updated after each instance
- `{savedir}/exp_dict.json` recording the ablation config

Pre-existing tree.json files under `--datastorm_run_dir` are reused; pass
`--no_skip_generated` to force regeneration.
