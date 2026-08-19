#!/usr/bin/env python3
"""
Salvage a *collapsed* run (tree.json present, but 0 is_final_selected → web-only
report) WITHOUT re-running curation.

Why it works: the collapse is caused by `--disable_followups` also disabling
node summarization (now fixed in tree_simulator.py), so the global-insight
selection ran blind on empty summaries and selected nothing — even though the
tree holds all the SQL results. We rebuild from the stored tree:

  1. Re-summarize: for every node that has a SQL result but no summary, call the
     same `summarize` LLM on its stored table snippet (no DB access needed).
  2. Re-select: rank the distinct-SQL nodes (rerank LLM if >max_insights) and set
     is_final_selected so the report uses the DB evidence.
  3. Write the updated tree.json (+ input.txt / warmstart copy) into a NEW run
     dir under datatalk/acled_ablations/<out_set>/<id>/, then generate the report
     + url_to_info.json there via the existing report salvage.

Usage:
  python acled_ablation/resummarize_salvage.py \
      --run-dir datatalk/acled_ablations/no_qc/8/<run> --id 8 --out-set no_qc_fixed
Run from repo root with the storm conda env.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "acled_ablation"))
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

from knowledge_storm.utils import load_api_key  # noqa: E402

load_api_key(toml_file_path=str(REPO / "secrets.toml"))

from knowledge_storm.langfuse_llm import get_llm, call_llm_with_structured_output  # noqa: E402
from knowledge_storm.datastorm.modules.tree_simulator import (  # noqa: E402
    SummarizeModel,
    RerankBestInsightsPreemptResponse,
)
import salvage_reports  # noqa: E402

MODEL = os.getenv("RESUMMARIZE_MODEL", "gpt-5")
MAX_INSIGHTS = 30
CONCURRENCY = 8


def walk(node: Dict[str, Any]):
    yield node
    for c in (node.get("children") or []):
        yield from walk(c)


def node_sql(node: Dict[str, Any]) -> str:
    sr = (node.get("dlg_turn") or {}).get("search_results") or []
    if not sr:
        return ""
    meta = sr[0].get("meta") or {}
    return (meta.get("preprocessed_sql") or meta.get("SQL") or "").strip()


def node_table(node: Dict[str, Any]) -> str:
    sr = (node.get("dlg_turn") or {}).get("search_results") or []
    snips = (sr[0].get("snippets") if sr else None) or []
    return snips[0] if snips else ""


async def resummarize(tree: Dict[str, Any]) -> int:
    """Summarize every node that has a SQL table but no summary (in place)."""
    today = datetime.now().strftime("%Y-%m-%d")
    llm = get_llm(model_name=MODEL)
    todo = [n for n in walk(tree) if node_table(n) and not (n.get("dlg_turn") or {}).get("summary")]
    sem = asyncio.Semaphore(CONCURRENCY)

    async def do(node):
        async with sem:
            try:
                res = await call_llm_with_structured_output(
                    "summarize",
                    {
                        "table_result": node_table(node),
                        "previous_observation": "No previous observation",
                        "today": today,
                    },
                    SummarizeModel,
                    llm,
                    langfuse_readonly=True,
                )
                node["dlg_turn"]["summary"] = (res.summary if res else "") or \
                    node["dlg_turn"].get("agent_utterance") or ""
            except Exception as exc:
                print(f"[warn] summarize failed for node {node.get('node_id')}: {exc}", file=sys.stderr)
                node["dlg_turn"]["summary"] = node["dlg_turn"].get("agent_utterance") or ""

    await asyncio.gather(*[do(n) for n in todo])
    return len(todo)


async def reselect(tree: Dict[str, Any], topic: str, db_desc: str, thesis: str) -> int:
    """Pick distinct-SQL evidence nodes and set is_final_selected. Uses the rerank
    LLM only if there are more distinct-SQL candidates than MAX_INSIGHTS."""
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for n in walk(tree):
        dlg = n.get("dlg_turn") or {}
        if not dlg.get("summary") or not node_table(n):
            continue
        sql = node_sql(n)
        if sql in seen:
            continue
        seen.add(sql)
        candidates.append(n)

    if not candidates:
        return 0

    if len(candidates) <= MAX_INSIGHTS:
        selected_ids = {n["node_id"] for n in candidates}
    else:
        rerank_input = {n["node_id"]: n["dlg_turn"]["summary"] for n in candidates}
        out = await call_llm_with_structured_output(
            "rerank_best_insights_preempt",
            {
                "max_num_insights": MAX_INSIGHTS,
                "topic": topic,
                "input": json.dumps(rerank_input, indent=2),
                "db_description": db_desc,
                "thesis": thesis,
            },
            RerankBestInsightsPreemptResponse,
            get_llm(model_name=MODEL),
            langfuse_readonly=True,
        )
        if out and getattr(out, "results", None):
            selected_ids = {r.node_id for r in out.results if r.node_id in rerank_input}
        else:
            selected_ids = set(list(rerank_input.keys())[:MAX_INSIGHTS])
        if not selected_ids:  # rerank returned nothing usable
            selected_ids = set(list(rerank_input.keys())[:MAX_INSIGHTS])

    n_sel = 0
    for n in walk(tree):
        if n.get("node_id") in selected_ids:
            n["is_final_selected"] = True
            n_sel += 1
    return n_sel


async def rebuild_tree(tree: Dict[str, Any], topic: str, db_desc: str, thesis: str) -> tuple:
    """Async: re-summarize nodes and re-select evidence (mutates tree in place)."""
    n_sum = await resummarize(tree)
    print(f"  re-summarized {n_sum} node(s)")
    n_sel = await reselect(tree, topic, db_desc, thesis)
    print(f"  selected {n_sel} evidence node(s)")
    return n_sum, n_sel


def run(run_dir: Path, eval_id: int, out_set: str) -> bool:
    tree_path = run_dir / "tree.json"
    if not tree_path.exists():
        print(f"[skip] no tree.json in {run_dir}")
        return False
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    topic = (run_dir / "input.txt").read_text(encoding="utf-8").strip() if (run_dir / "input.txt").exists() else ""
    thesis = (tree.get("thesis") or "").strip()
    db_desc = salvage_reports._db_description("acled")

    print(f"[resummarize] {run_dir.name}")
    # Async stage (its own event loop); must finish before the report generation,
    # which runs its own asyncio.run() internally and cannot be nested.
    _, n_sel = asyncio.run(rebuild_tree(tree, topic, db_desc, thesis))
    if n_sel == 0:
        print("  ERROR: still 0 selected — aborting.")
        return False

    # Write the rebuilt run dir under the new set.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = run_dir.name.split("__")[0]
    out_dir = REPO / "datatalk" / "acled_ablations" / out_set / str(eval_id) / f"{slug}__resummarized_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tree.json").write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    for fn in ("input.txt", "warmstart_conversation.json", "run_metadata.json"):
        if (run_dir / fn).exists():
            shutil.copy2(run_dir / fn, out_dir / fn)
    print(f"  wrote rebuilt tree -> {out_dir}")

    # Generate the report + url_to_info (synchronous; manages its own event loop).
    ok = salvage_reports.salvage_run(out_dir)
    if ok:
        print(f"  REPORT generated in {out_dir}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--out-set", default="no_qc_fixed")
    args = ap.parse_args()
    ok = run(Path(args.run_dir).resolve(), args.id, args.out_set)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
