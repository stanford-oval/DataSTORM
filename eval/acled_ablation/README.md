# ACLED ablation runner

Drives component-level ablations of DataSTORM on the ACLED benchmark (paper §4.2),
matching the published baseline config:

| Parameter | Value |
|-----------|-------|
| `first_level_questions` | 8 |
| `max_tree_depth` | 8 |
| `each_level_population_control_num` | 3 |
| `max_global_insights` | 30 |
| `expansion_max_questions` | 5 |
| `warmstart_max_num_experts` / `warmstart_max_turn_per_experts` | 3 / 2 (orchestrator defaults) |
| Co-STORM warm-start | **ON** |
| Final 5-stage editorial pipeline | **ON** |
| `costorm_retriever` | serper (date-restricted per topic) |
| `generation_module_model` | gpt-5.1 (the paper's editorial model) |
| `datatalk_engine` / `datastorm_main_model` | gpt-5 |

Local-only overrides (vs the paper hosts): `--disable_upload_to_azure` (no azcopy)
and `--disable_graphs` (avoids the docker matplotlib sidecar — graphs are cosmetic
and not part of any reported metric).

## Setup

```bash
export DATASTORM_REPO_ROOT=/path/to/datastorm   # default: the repo root
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=...                        # any OpenAI-compatible endpoint
export SERPER_API_KEY=...                                # for Co-STORM warm-start retrieval
```

PostgreSQL must have an `acled` database with the ACLED events table, accessible via
`select_user` (see `knowledge_storm/datatalk_agent` + `$DATATALK_DOMAINS_ROOT/acled/`).
`db_details.json` must have `"suql_enabled": false` to skip the SUQL research-preview path.

## Ablations

| `--ablation` | CLI flag added | Tests |
|--------------|----------------|-------|
| `baseline` | (none) | Reference config |
| `no_qc` | `--disable_followups` | Query consistency module |
| `no_inductive` | `--no_summary_stats` | Bottom-up summary stats |
| `no_thesis` | `--skip_thesis` | Thesis-driven exploration |
| `no_warm_start` | `--no_warm_start` | Co-STORM warm-start phase |

## Example

```bash
# Single ablation × 5 IDs (sequential within the process)
python acled_ablation/run_acled_ablation.py \
    --ablation no_qc \
    --ids 1 6 8 14 15 \
    --datastorm_run_dir /path/to/runs/no_qc

# Parallelize across ablations: launch each as its own process with a
# different --datastorm_run_dir.
```

## Topic mapping

See `topics.json` — IDs 1–20 with topic strings and the per-topic
`serper_query_params.tbs` date restriction (matches the paper's reference
articles' publication dates, prevents leakage of post-hoc commentary).

## Output

Each run produces (under `{datastorm_run_dir}/{id}/{slug}__{ts}/`):
- `tree.json` — the search tree
- `co_storm_report*.txt` / `*.md` — the staged editorial pipeline output
- `run_metadata.json` — recorded config
- intermediate JSONs from warm-start, conversation log, etc.

Scoring is **post-hoc** against the paper metrics, via the existing eval
pipelines in `breakdown_evaluator/` (DB-use),
`consistency_eval/`, and the criteria-match / RACE eval setup.
