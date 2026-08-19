"""Export this project's prompts from Langfuse into `prompts/` as JSON.

The pipeline normally fetches every prompt from Langfuse at runtime, which makes
the repo unrunnable for anyone without access to our Langfuse project. This
script snapshots the prompts so the released code can fall back to local files
(see knowledge_storm/langfuse_prompts.py).

Usage:
    python scripts/export_langfuse_prompts.py             # export production label
    python scripts/export_langfuse_prompts.py --label latest

Requires LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY / LANGFUSE_HOST in the
environment or in secrets.toml at the repo root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Prompts belonging to the DataSTORM pipeline and its evaluation harnesses.
# Other prompts live in the same Langfuse project but belong to unrelated lab
# work (spinach_*, paper_classifier, repo_audit_*, generate_benchmark_manifest)
# and are deliberately excluded from the public release.
PIPELINE_PROMPTS = [
    # --- tree search / knowledge curation ---
    "starting_questions",
    "expand_node_breakdown",
    "expand_node_exploration_direct_sql_gen",
    "expand_node_exploration_question_gen",
    "followups_to_clean_one_level",
    "rerank_best_insights_preempt",
    "final_insights_consolidation",
    "summarize",
    "plot_interpretation",
    "graph_generation",
    "graph_generation_matplotlib",
    "datastorm_determine_problematic_queries",
    # --- thesis / critique ---
    "generate_thesis",
    "refine_thesis",
    "article_critique_impact_analysis",
    "article_critique_resynthesis",
    "fact_check_sentence",
    # --- staged report generation ---
    "staged_report_plan",
    "staged_report_title",
    "staged_report_section_draft",
    "staged_report_section_revise",
    "staged_report_evidence_note",
    "staged_report_final_polish",
    # --- datatalk SQL agent ---
    "datatalk_controller",
    "datatalk_reporter",
    "datatalk_select_entities",
    "datatalk_verify_domain_specific_instructions",
    # --- evaluation harnesses ---
    "datastorm_acled_eval_breakdown_prompt",
    "datastorm_acled_eval_data_source_prompt",
    "datastorm_acled_eval_insight_match_prompt",
    "acled_eval_criteria_generate",
    "acled_eval_criteria_grade",
    "acled_eval_thesis_match",
    "drb_eval_criteria_comprehensiveness_en",
    "drb_eval_criteria_insight_en",
    "drb_eval_criteria_instruction_following_en",
    "drb_eval_criteria_readability_en",
    "drb_eval_dimension_weight_en",
    "drb_eval_merged_score_en",
]



def load_secrets() -> None:
    secrets = REPO_ROOT / "secrets.toml"
    if secrets.exists():
        from knowledge_storm.utils import load_api_key

        load_api_key(str(secrets))


def serialise(prompt) -> dict:
    """Flatten a Langfuse prompt object into a JSON-round-trippable dict."""
    is_chat = getattr(prompt, "type", None) == "chat" or isinstance(
        getattr(prompt, "prompt", None), list
    )
    return {
        "name": prompt.name,
        "version": getattr(prompt, "version", None),
        "type": "chat" if is_chat else "text",
        "labels": list(getattr(prompt, "labels", []) or []),
        "tags": list(getattr(prompt, "tags", []) or []),
        "config": getattr(prompt, "config", None) or {},
        "prompt": prompt.prompt,
    }


def export(names: list[str], out_dir: Path, label: str) -> tuple[int, list[str]]:
    from langfuse import get_client

    client = get_client()
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []
    for name in names:
        try:
            prompt = client.get_prompt(name, label=label, cache_ttl_seconds=0)
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__} {exc}")
            continue
        payload = serialise(prompt)
        (out_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        ok += 1
        print(f"  {name:<52} v{payload['version']} ({payload['type']})")
    return ok, failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="production", help="Langfuse label to export")
    ap.add_argument("--out", default=str(REPO_ROOT / "prompts"))
    args = ap.parse_args()

    load_secrets()
    out = Path(args.out)

    print(f"Exporting pipeline prompts (label={args.label}) -> {out}")
    ok, failed = export(PIPELINE_PROMPTS, out, args.label)

    manifest = {
        "label": args.label,
        "pipeline_prompts": ok,
        "failures": failed,
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nExported {ok} prompts")
    if failed:
        print("FAILURES:")
        for f in failed:
            print("  ", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
