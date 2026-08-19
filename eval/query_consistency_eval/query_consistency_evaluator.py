#!/usr/bin/env python3
"""
Query-consistency evaluator (Tier 1, LLM-judge).

Measures whether the SQL queries grounding a single generated report are
mutually *consistent* — i.e. the same entity / place / time-window / concept is
filtered the same way across queries, so the resulting numbers are comparable.
This is exactly what the DataSTORM follow-up "QC" step is meant to standardize.

One gpt-5 call per report: we feed all cited (NL-question, SQL, result-preview)
triples and ask the judge to group filter predicates by concept, rate intra-
group consistency per dimension, and emit flagged conflicts + a 0-1 score.

Input is the report's run directory (needs url_to_info.json — the SQL sources
are read from there, which is robust to citation format). Run from repo root
with the storm conda env.

Usage:
  python query_consistency_eval/query_consistency_evaluator.py \
      --report-dir datatalk/acled_ablations/no_qc/9/<run> \
      --output-file query_consistency_eval/results/no_qc/9.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

from knowledge_storm.utils import load_api_key  # noqa: E402

load_api_key(toml_file_path=str(REPO / "secrets.toml"))

from langchain_openai import ChatOpenAI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

DEFAULT_MODEL = os.getenv("QC_EVAL_MODEL", "gpt-5")
MAX_SQL_CHARS = 1600
# Short preview of each query's returned table, just to convey scope. Set
# QC_MAX_RESULT_CHARS=0 to OMIT result tables entirely — needed for violence-heavy
# topics whose result snippets (casualty notes) trip Azure's content filter;
# consistency is judged on the SQL predicates, so omitting the data is fine.
MAX_RESULT_CHARS = int(os.getenv("QC_MAX_RESULT_CHARS", "700"))

DIMENSIONS = [
    "actor_entity",        # actor/group string matching (LIKE vs ILIKE ANY, exclusions, assoc_actor cols)
    "geography",           # admin1/admin2/country/region normalization for the same place
    "temporal",            # date windows for the same period (BETWEEN vs year IN, bounds)
    "event_taxonomy",      # event_type / sub_event_type / disorder_type for the same concept
    "counting_aggregation",# COUNT(*) vs COUNT(DISTINCT), fatalities SUM, geo_precision handling
    "population_scope",    # base set: country/region restriction present in one query, absent in another
]


# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #
class Conflict(BaseModel):
    dimension: str = Field(description=f"One of: {', '.join(DIMENSIONS)}.")
    concept: str = Field(description="The entity/place/time-window/concept the conflicting filters both target, e.g. 'Al Jazirah state' or 'November 2024'.")
    query_indices: List[int] = Field(description="The 0-based indices (as labeled Q0, Q1, ...) of the conflicting queries.")
    predicate_a: str = Field(description="The relevant filter predicate from the first query.")
    predicate_b: str = Field(description="The differing filter predicate from the second query.")
    severity: str = Field(description="'result-changing' if the two predicates would return different rows for the same concept, else 'cosmetic'.")
    explanation: str = Field(description="Why these are inconsistent and the likely effect on the numbers.")


class DimensionScore(BaseModel):
    dimension: str
    score: float = Field(description="0.0-1.0 consistency for this dimension (1.0 = every multi-query concept group is filtered identically; lower as result-changing conflicts appear). 1.0 if the dimension has no multi-query groups.")
    num_concept_groups: int = Field(description="Number of concepts in this dimension that are referenced by 2+ queries (the groups consistency is judged over).")
    num_consistent_groups: int = Field(description="How many of those groups are internally consistent.")
    notes: str = Field(default="")


class ConsistencyReport(BaseModel):
    per_dimension: List[DimensionScore]
    conflicts: List[Conflict]
    overall_score: float = Field(description="0.0-1.0 overall query consistency, the mean of the per-dimension scores.")
    summary: str = Field(description="2-3 sentence summary of the report's query consistency.")


SYSTEM_PROMPT = (
    "You are a meticulous SQL auditor evaluating the INTERNAL CONSISTENCY of the "
    "set of SQL queries that ground a single analytical report over the ACLED "
    "conflict-events database. The report makes quantitative claims, each backed "
    "by a SQL query. For the numbers to be trustworthy and mutually comparable, "
    "queries that refer to the SAME entity, place, time-window, or concept must "
    "filter it the SAME way.\n\n"
    "You will be given all the report's SQL queries (each with its natural-"
    "language question and a short preview of the rows it returned). Audit their "
    "consistency along these dimensions:\n"
    "  - actor_entity: actor/group matching (e.g. actor1 LIKE '%RDF%' vs "
    "COALESCE(...) ILIKE ANY('%RDF%','%Rwanda Defence Force%') AND NOT ILIKE "
    "'%EPRDF%'); which actor columns are searched; false-positive exclusions.\n"
    "  - geography: normalization of the same place (admin1 = 'Al Jazirah' vs "
    "admin1 ILIKE ANY('Al Jazirah','Al-Jazirah','Gezira',...)); granularity.\n"
    "  - temporal: the same period expressed with the same bounds (event_date "
    "BETWEEN ... vs year IN (...)).\n"
    "  - event_taxonomy: the same concept defined with the same event_type / "
    "sub_event_type / disorder_type values.\n"
    "  - counting_aggregation: COUNT(*) vs COUNT(DISTINCT ...); fatalities SUM; "
    "geo_precision / duplicate handling.\n"
    "  - population_scope: the base set (e.g. one query adds country='Sudan', "
    "another omits it for the same regional claim).\n\n"
    "Method: (1) Across the queries, find CONCEPT GROUPS — sets of 2+ queries "
    "that target the same entity/place/period/concept within a dimension. Ignore "
    "purely structural predicates (join conditions like a.admin1 = b.admin1). "
    "(2) For each group, decide whether the filters are semantically equivalent. "
    "(3) Score each dimension 0-1 = fraction of its multi-query groups that are "
    "consistent (1.0 if it has no multi-query groups). (4) List the conflicts, "
    "marking severity 'result-changing' if the predicates would return different "
    "rows for the same concept, else 'cosmetic'. overall_score = mean of the "
    "per-dimension scores. Be precise; only flag genuine inconsistencies, not "
    "legitimately different filters for genuinely different concepts."
)


def truncate_table(md: str, limit: int) -> str:
    if limit <= 0:
        return "(result table omitted)"
    if not md:
        return "(empty)"
    lines = md.splitlines()
    if len(lines) > 10:
        lines = lines[:10] + ["| ... |"]
    out = "\n".join(lines)
    return out[:limit]


def extract_sql_triples(report_dir: Path) -> List[Dict[str, str]]:
    info_path = report_dir / "url_to_info.json"
    if not info_path.exists():
        return []
    data = json.loads(info_path.read_text(encoding="utf-8"))
    uti = data.get("url_to_info", {}) or {}
    triples: List[Dict[str, str]] = []
    seen_sql = set()
    for v in uti.values():
        meta = v.get("meta") or {}
        if str(meta.get("designation", "")).upper() != "SQL":
            continue
        sql = (meta.get("preprocessed_sql") or meta.get("SQL") or "").strip()
        if not sql or sql in seen_sql:
            continue
        seen_sql.add(sql)
        triples.append({
            "nl_question": (meta.get("query") or "").strip(),
            "sql": sql[:MAX_SQL_CHARS],
            "result_preview": truncate_table(meta.get("sql_result") or "", MAX_RESULT_CHARS),
        })
    return triples


def build_user_prompt(triples: List[Dict[str, str]]) -> str:
    parts = [f"The report is grounded by {len(triples)} SQL queries:\n"]
    for i, t in enumerate(triples):
        parts.append(
            f"=== Q{i} ===\n"
            f"Question: {t['nl_question'] or '(none)'}\n"
            f"SQL:\n{t['sql']}\n"
            f"Returned (preview):\n{t['result_preview']}\n"
        )
    parts.append(
        "\nAudit the internal consistency of these queries per the instructions "
        "and return the structured result."
    )
    return "\n".join(parts)


def make_llm(model: str) -> ChatOpenAI:
    base = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/"
    return ChatOpenAI(
        openai_api_base=base,
        model=model,
        temperature=1.0 if "gpt-5" in model else 0.0,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


def evaluate(report_dir: Path, model: str) -> Dict[str, Any]:
    triples = extract_sql_triples(report_dir)
    base = {"report_dir": str(report_dir), "model": model, "num_queries": len(triples)}
    if len(triples) < 2:
        base.update({
            "overall_score": None,
            "applicable": False,
            "note": "Fewer than 2 SQL queries — query consistency is not applicable.",
        })
        return base
    llm = make_llm(model).with_structured_output(ConsistencyReport)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(triples)},
    ]
    result = llm.invoke(messages)
    out = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    out.update(base)
    out["applicable"] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-dir", required=True, help="Run directory containing url_to_info.json.")
    ap.add_argument("--output-file", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    result = evaluate(Path(args.report_dir).resolve(), args.model)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output_file:
        p = Path(args.output_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(payload + "\n", encoding="utf-8")
        sc = result.get("overall_score")
        print(f"[qc] {args.report_dir} -> overall={sc} ({result.get('num_queries')} queries) -> {p}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
