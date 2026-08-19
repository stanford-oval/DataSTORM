# DataSTORM

**Deep Research on Large-Scale Databases using Exploratory Data Analysis and
Data Storytelling**

[![arXiv](https://img.shields.io/badge/arXiv-2604.06474-b31b1b.svg)](https://arxiv.org/abs/2604.06474)

DataSTORM performs exploratory data analysis over large databases and writes a
grounded, analyst-style report about what it found.

Given a research question, it runs a tree search: each node poses a sub-question,
answers it by generating and executing SQL against the database, judges whether
the result is interesting, and expands promising branches further. Findings are
reranked into a set of global insights, and a staged generation pipeline turns
those into a report where every claim is traceable to the query that produced it.
Web search can be mixed in so that database findings are contextualised against
external reporting.

---

## How it works

Two components:

**`knowledge_storm/datatalk_agent/`** — the SQL agent. Given a natural-language
question it inspects the schema, resolves entity mentions against the database,
generates SQL (optionally SUQL, which lets a query filter on free-text columns
semantically), executes it, and summarises the result. It runs in-process; there
is no separate service to start.

**`knowledge_storm/datastorm/`** — the research pipeline. Drives the agent across
a search tree, scores and prunes branches, consolidates insights, and generates
the final report. `datatalk_basic_article_generation_costorm.py` is the entry
point and also exposes a FastAPI mode (`--api_mode`).

---

## Install

Prerequisites: Python 3.11, a **PostgreSQL server** (14+) reachable from the
machine running DataSTORM, and — only if you use `--enable_python` — Docker for
the sandboxed analysis step.

```bash
conda create -n datastorm python=3.11
conda activate datastorm
pip install -r requirements.txt
```

`psycopg2-binary` ships the Postgres client, so no separate `libpq` install is
needed.

## Configure

```bash
cp secrets_example.toml secrets.toml
```

At minimum set `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` (any
OpenAI-compatible endpoint works) and `SERPER_API_KEY` for web search. Every
other key is optional and documented inline in `secrets_example.toml`.

## Per-domain data

The SQL agent needs a prepared directory per domain, under
`DATATALK_DOMAINS_ROOT` (default `<repo>/datatalk_domains`):

```
datatalk_domains/<domain>/
  db_details.json                    connection config: db_type, database_name,
                                     db_path, db_secrets_file, suql_enabled, ...
  _lookup_table.csv                  entity → table lookup for entity linking
  domain_specific_instructions.csv   guidance injected into SQL generation
```

A minimal `db_details.json` for a PostgreSQL domain:

```json
{
  "db_type": "postgres",
  "database_name": "acled",
  "db_secrets_file": "/path/to/db_secrets.json"
}
```

Connections are made with `psycopg2` and default to `127.0.0.1:5432` with the
role `select_user`, a 30-second statement timeout, and an automatic `LIMIT 1000`
appended to every query. Point `db_secrets_file` at your own credentials to
change host, port or role.

> **Use a read-only role.** The agent generates SQL from an LLM and executes it
> directly against your database. The default `select_user` name reflects the
> intent: grant `SELECT` only, on only the tables the agent should see. Nothing
> in the pipeline prevents a generated statement from being destructive if the
> role it runs as is permitted to be.

That is everything needed to run. [SUQL](#optional-suql) and the other
integrations below are optional.

## Quickstart

Smoke test — depth 3, roughly five minutes:

```bash
python datatalk_basic_article_generation_costorm.py \
    --topic "Do a deep research on Petro’s presidency in Columbia and Columbia’s conflict summary and outlook, as if we are at the end of 2024 heading into 2025." \
    --domain acled \
    --output_dir ./runs/smoke \
    --first_level_questions 2 \
    --max_tree_depth 3 \
    --each_level_population_control_num 2 \
    --max_global_insights 5 \
    --expansion_max_questions 2 \
    --disable_graphs \
    --disable_upload_to_azure
```

Full run, matching the published ACLED configuration:

```bash
python datatalk_basic_article_generation_costorm.py \
    --topic "Do a deep research on Petro’s presidency in Columbia and Columbia’s conflict summary and outlook, as if we are at the end of 2024 heading into 2025." \
    --domain acled \
    --output_dir ./runs/acled \
    --serper_query_params '{"tbs": "cdr:1,cd_max:12/12/2024"}' \
    --first_level_questions 8 \
    --max_tree_depth 8 \
    --each_level_population_control_num 3 \
    --max_global_insights 30 \
    --expansion_max_questions 5
```

`--serper_query_params` constrains web search to before the cutoff date so the
run cannot see reporting from after the period it analyses.

### Frequently used options

| flag | meaning |
|---|---|
| `--domain` | which directory under `DATATALK_DOMAINS_ROOT` to query |
| `--first_level_questions` | breadth of the first tree level |
| `--max_tree_depth` | maximum search depth |
| `--each_level_population_control_num` | branches kept per level |
| `--expansion_max_questions` | sub-questions generated per expansion |
| `--max_global_insights` | insights carried into report generation |
| `--datastorm_main_model` | model for the search pipeline |
| `--generation_module_model` | model for report generation |
| `--enable_python` | allow sandboxed Python analysis (needs Docker) |
| `--disable_graphs` | skip figure generation |
| `--skip_final_article` | stop after the search, no report |
| `--critique` | run the critique-and-revise pass |
| `--api_mode` / `--port` | serve `/generate_article` over HTTP |

`python datatalk_basic_article_generation_costorm.py --help` lists all 34.

## Optional: SUQL

[SUQL](https://github.com/stanford-oval/suql) lets a query filter on free-text
columns semantically, so the agent can search structured and unstructured
fields in one statement. Follow that repo for set-up; then enable it per domain
in `db_details.json`:

```json
{
  "db_type": "postgres",
  "database_name": "acled",
  "db_secrets_file": "/path/to/db_secrets.json",
  "suql_enabled": true,
  "table_w_ids": { "events": "event_id_cnty" }
}
```

`table_w_ids` maps each table to its primary-key column. SUQL needs one to tie a
free-text match back to a row, and the lookup CSV does not always carry it — the
ACLED domain points `events` at `event_id_cnty`. Everything else is the ordinary
connection config, so a SUQL domain is the minimal file plus these two keys.

With `suql_enabled: true`, both SUQL servers must be running before a query is
issued — the embedding server (`SUQL_EMBEDDING_SERVER`, default
`127.0.0.1:8505`) and the free-text server (`SUQL_FREE_TEXT_SERVER`, default
`127.0.0.1:8510`). SUQL also tightens the automatic row limit from 1000 to 10,
since each returned row may carry an embedding lookup.

**SUQL is PostgreSQL-only.**

## Optional: other integrations

- **Redis** (`redis://localhost:6379`) caches entity linking. Absent is fine;
  the code degrades gracefully.
- **Azure Blob Storage** (`AZURE_SAS_TOKEN`, `AZURE_STORAGE_DEST`) uploads
  result CSVs and cites them by URL. Pass `--disable_upload_to_azure` to skip.
- **SQLite** works for small single-file domains (`"db_type": "sqlite"` with a
  `db_path`), but without SUQL. PostgreSQL and SQLite are the only supported
  backends.

## Paper results

The runs reported in [the paper](https://arxiv.org/abs/2604.06474) are published
in [`results/`](results/README.md), one directory per table:

| result | paper |
|---|---|
| `results/insight_bench/` — 100 tasks, DataSTORM and the AgentPoirot baseline | Table 1 |
| `results/acled_benchmark/criteria_match/` + `race/` — 20 topics × 3 systems | Table 2 |
| `results/acled_benchmark/ablations/` — leave-one-out arms | Table 4a |
| `eval/query_consistency_eval/results/` — query-consistency ablation | Table 4b |
| `results/acled_benchmark/rubrics/` — reference-induced rubrics | Table 7 |
| `eval/counterevidence_audit/outputs/` — thesis-refinement evidence audit | Appendix E.2 |

[`results/README.md`](results/README.md) documents how each number was computed
and which JSON field backs every cell.

> **Data disclaimer.** The ACLED benchmark is built on data and publications from
> the [Armed Conflict Location & Event Data Project](https://acleddata.com),
> which we cannot redistribute. Three things are therefore **not** in this
> repository:
>
> - the **ACLED event database** the agent queries;
> - the **20 ACLED articles** used as evaluation references — listed by exact URL
>   in [`eval/deep_research_bench_evaluator/references/README.md`](eval/deep_research_bench_evaluator/references/README.md)
>   so you can retrieve them yourself;
> - the **generated ACLED reports and exploration trees**.
>
> Obtain the data and articles directly from ACLED under
> [their terms of use](https://acleddata.com/terms-of-use/), and cite ACLED in
> any derived work. These materials are governed by ACLED's terms, not by this
> repository's Apache-2.0 license.

Regenerate `results/` with `python scripts/export_results.py`.

## Evaluation

`eval/` holds the harnesses used in the paper:

| directory | what it measures |
|---|---|
| `breakdown_evaluator/` | database-use ratio (Table 2) |
| `deep_research_bench_evaluator/` | DeepResearch-Bench criteria scoring |
| `query_consistency_eval/` | consistency of repeated queries |
| `counterevidence_audit/` | whether counterevidence is surfaced |
| `acled_ablation/`, `insight_bench_ablation/` | ablation drivers |

Most evaluators read full run directories; point `DATASTORM_ACLED_RUNS` at them.

## Repository layout

```
knowledge_storm/          core library
  datastorm/              tree search, insight ranking, report generation
  datatalk_agent/         NL → SQL/SUQL agent
  collaborative_storm/    inherited Co-STORM modules
prompts/                  prompt snapshot (JSON)
scripts/                  prompt export/push, results export
eval/                     evaluation harnesses
results/                  published reproducibility artifacts
datatalk_basic_article_generation_costorm.py    entry point
```

## Citation

```bibtex
@misc{liu2026datastormdeepresearchlargescale,
      title={DataSTORM: Deep Research on Large-Scale Databases using Exploratory Data Analysis and Data Storytelling},
      author={Shicheng Liu and Yucheng Jiang and Sajid Farook and Camila Nicollier Sanchez and David Fernando Castro Pena and Monica S. Lam},
      year={2026},
      eprint={2604.06474},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.06474},
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Published artifacts under `results/` are subject to the terms of their
underlying data sources rather than this license; see
[`results/README.md`](results/README.md).
