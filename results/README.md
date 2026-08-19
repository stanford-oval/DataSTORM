# Reproducibility artifacts

Published outputs from the two evaluations in the paper. Regenerate this
directory with `python scripts/export_results.py`.

## `acled_benchmark/` — the ACLED evaluation

Everything backing the ACLED results in the paper. Each subdirectory maps to one
table.

```
acled_benchmark/rubrics/          <id>_criteria.json               Table 7
acled_benchmark/criteria_match/   <id>_criteria_match_<system>.json Table 2, Ref-Induced
acled_benchmark/race/             <id>_<system>.json                Table 2, RACE columns
acled_benchmark/ablations/        <arm>/<id>_{criteria,race,datasource}.json  Table 4a
```

`<system>` is one of `costorm` (DataSTORM), `openai_dr` (OpenAI DR via MCP), or
`openai_dr_csv` (OpenAI DR over CSV). All 20 topics per system.

**Rubrics.** The benchmark scores a report against a human-written ACLED analysis
of the same situation. Rather than compare texts directly, we induce a rubric
from the reference article — named criteria describing what a good analysis must
cover — and grade reports against it. The rubrics contain no ACLED text, so they
are published in full; article IDs match
[`eval/deep_research_bench_evaluator/references/README.md`](../eval/deep_research_bench_evaluator/references/README.md),
which lists the source URL for every reference article.

**Reproducing Table 2.** Mean `overall_score` over the 20 files per system:

| system | Ref-Induced (`criteria_match/`) | RACE overall (`race/`) |
|---|---|---|
| OpenAI DR (MCP) | 48.5% | 46.1 |
| OpenAI DR (CSV) | 51.2% | 46.8 |
| DataSTORM | **61.8%** | **52.6** |

`race/` files also carry the four RACE dimensions — `comprehensiveness`,
`insight`, `instruction_following`, `readability` — as fractions; multiply by 100
for the paper's scale.

**Reproducing Table 4a.** `ablations/<arm>/` holds one directory per leave-one-out
arm, where the arm name is the flag passed to the pipeline:
`no_thesis` = `--skip_thesis`, `no_inductive` = `--no_summary_stats`,
`no_qc` = `--disable_followups`.

### How to get the ACLED data

The ACLED event data is not redistributed here. We obtained it under the
**Research** access tier in myACLED, downloading the regional event files from
<https://acleddata.com/conflict-data/download-data-files> (accessed
01/02/2026). Register for the same tier to retrieve the equivalent files.

## `insight_bench/` — 100 analytics tasks

DataSTORM on [InsightBench](https://github.com/ServiceNow/insight-bench),
which scores how many of a dataset's ground-truth insights an agent recovers.

```
insight_bench/scores/pred_gt_<n>.json           DataSTORM: predictions + scores
insight_bench/baseline_scores/pred_gt_<n>.json  InsightBench agent baseline
```

Each file holds our predicted insights and summary plus every judge's score.
**InsightBench's own ground-truth insights and summaries are stripped** — those
are the benchmark's to distribute, not ours. Re-attach them from the upstream
benchmark if you want to recompute scores from scratch.

### Headline numbers

DataSTORM is run `20260325_130410` (gpt-5); the baseline is the InsightBench
agent, run `baseline-20260331-034919` (gpt-5). Both cover all 100 datasets.

All figures are **LLM-judge (G-Eval) scores**.

| metric | judge | DataSTORM | baseline | JSON field (DataSTORM) | JSON field (baseline) |
|---|---|---|---|---|---|
| Insight recall | gpt-4o | **0.6187** | 0.4708 | `score_insights` | `score_insights` |
| Insight recall | Qwen3-30B | **0.6913** | 0.4994 | `score_insights_Qwen3-30B-A3B-Instruct-2507` | `score_insights_Qwen3-30B-A3B-Instruct-2507` |
| Summary quality | gpt-4o | **0.5248** | 0.4661 | `score_llm_predicted_summary` | `score_summary` |
| Summary quality | Qwen3-30B | **0.5869** | 0.5152 | `score_llm_predicted_summary_Qwen3-30B-A3B-Instruct-2507` | `score_predicted_summary_Qwen3-30B-A3B-Instruct-2507` |

Every value is the mean over the 100 per-dataset files:

* DataSTORM — `insight_bench/scores/pred_gt_*.json`
* baseline — `insight_bench/baseline_scores/pred_gt_*.json`
