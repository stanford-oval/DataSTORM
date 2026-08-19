#!/usr/bin/env python3
"""Audit whether final ACLED articles include evidence that challenges early theses.

The audit uses the git-tracked ACLED run directories. For each submitted run it
loads:

  - tree.json: initial and refined thesis events
  - staged_report_notes.json: evidence selected for the final staged article
  - staged_report_provenance.json: final thesis metadata

GPT-5 labels each included evidence item relative to both the initial thesis and
the final thesis. This gives a concrete first-pass measure of whether the final
submitted articles included qualifying or refuting evidence against the early
working hypothesis.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402


STANCE = Literal["supports", "qualifies", "refutes", "mixed", "unrelated"]
VALIDITY = Literal["valid", "invalid", "unclear"]
MATERIALITY = Literal["material", "minor"]
COUNTER_TYPE = Literal[
    "scope_limit",
    "exception",
    "direct_contradiction",
    "alternative_mechanism",
    "measurement_caveat",
    "none",
]


class EvidenceJudgment(BaseModel):
    evidence_id: int
    validity: VALIDITY = Field(
        description="Whether the evidence text appears usable for judging the thesis."
    )
    materiality: MATERIALITY = Field(
        description="Material if it affects the central thesis, not just background."
    )
    stance_vs_initial: STANCE = Field(
        description="How the evidence relates to the initial generated thesis."
    )
    stance_vs_final: STANCE = Field(
        description="How the evidence relates to the final thesis used in the article."
    )
    counterevidence_type: COUNTER_TYPE = Field(
        description="Type of thesis challenge if stance_vs_initial qualifies/refutes/mixed."
    )
    rationale: str = Field(description="One concise sentence explaining the labels.")
    quoted_basis: str = Field(
        description="Short quote or paraphrase from the evidence that justifies the label."
    )


class TopicJudgments(BaseModel):
    judgments: list[EvidenceJudgment]


def load_secrets(path: Path) -> None:
    """Load API keys from secrets.toml without importing the full app stack."""
    if not path.exists():
        return
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
        import tomli as tomllib  # type: ignore

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    for key, value in data.items():
        if isinstance(value, str):
            os.environ[key] = value


load_secrets(PROJECT_ROOT / "secrets.toml")


def tracked_files(root: Path) -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "datatalk/acled"], cwd=root
    )
    return [
        root / p
        for p in raw.decode("utf-8", errors="surrogateescape").split("\0")
        if p
    ]


def clip(text: Any, limit: int) -> str:
    value = "" if text is None else str(text)
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 20].rstrip() + " ...[truncated]"


SANITIZE_REPLACEMENTS = [
    (r"\bviolence against civilians\b", "civilian-harm incidents"),
    (r"\bcivilian targeting\b", "civilian-harm pattern"),
    (r"\btargeting civilians\b", "affecting civilians"),
    (r"\bviolence\b", "conflict activity"),
    (r"\bviolent\b", "conflict-related"),
    (r"\bfatalities\b", "casualty counts"),
    (r"\bfatality\b", "casualty count"),
    (r"\bdeaths\b", "casualty counts"),
    (r"\bdeadlier\b", "higher-casualty"),
    (r"\bkilled\b", "associated with casualties"),
    (r"\bkillings\b", "casualty incidents"),
    (r"\bmassacres?\b", "severe incidents"),
    (r"\bsuicide blasts?\b", "high-profile explosive incidents"),
    (r"\bsuicide bombings?\b", "high-profile explosive incidents"),
    (r"\bbombings?\b", "explosive incidents"),
    (r"\bair/drone strikes?\b", "remote-strike incidents"),
    (r"\bairstrikes?\b", "remote-strike incidents"),
    (r"\bshelling/artillery/missile attacks?\b", "remote-fire incidents"),
    (r"\battacks?\b", "incidents"),
    (r"\bassaults?\b", "incidents"),
]


def sanitize_for_policy(text: Any) -> str:
    """Neutralize conflict wording before sending text to Azure content filters."""
    value = "" if text is None else str(text)
    for pattern, replacement in SANITIZE_REPLACEMENTS:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_runs(root: Path) -> list[dict[str, Any]]:
    files = tracked_files(root)
    tree_paths = sorted(p for p in files if p.name == "tree.json")
    runs = []

    for tree_path in tree_paths:
        run_dir = tree_path.parent
        notes_path = run_dir / "staged_report_notes.json"
        provenance_path = run_dir / "staged_report_provenance.json"
        if not notes_path.exists() or not provenance_path.exists():
            continue

        tree = load_json(tree_path)
        notes = load_json(notes_path)
        provenance = load_json(provenance_path)
        thesis_events = sorted(
            tree.get("thesis_events", []), key=lambda event: event.get("depth", 0)
        )
        if not thesis_events:
            continue

        initial = thesis_events[0].get("selected") or {}
        final = thesis_events[-1].get("selected") or {}
        topic_id = tree_path.relative_to(root).parts[2]

        runs.append(
            {
                "topic_id": int(topic_id) if topic_id.isdigit() else topic_id,
                "run_dir": run_dir,
                "tree_path": tree_path,
                "notes_path": notes_path,
                "provenance_path": provenance_path,
                "topic": provenance.get("topic") or "",
                "initial_thesis_depth": thesis_events[0].get("depth"),
                "initial_thesis": initial.get("thesis") or "",
                "initial_research_strategy": initial.get("research_strategy") or "",
                "final_thesis": provenance.get("thesis")
                or final.get("thesis")
                or "",
                "notes": notes,
            }
        )
    return runs


def evidence_payload(run: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for raw in run["notes"]:
        evidence = raw.get("evidence") or {}
        evidence_id = evidence.get("evidence_id")
        if evidence_id is None:
            continue
        items.append(
            {
                "evidence_id": int(evidence_id),
                "depth": evidence.get("depth"),
                "post_initial_thesis": (
                    isinstance(evidence.get("depth"), int)
                    and isinstance(run.get("initial_thesis_depth"), int)
                    and evidence.get("depth") > run["initial_thesis_depth"]
                ),
                "source_type": evidence.get("source_type"),
                "question": clip(evidence.get("question"), 260),
                "finding": clip(evidence.get("finding"), 900),
            }
        )
    return items


def make_llm(model: str) -> ChatOpenAI:
    base = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    kwargs: dict[str, Any] = {
        "base_url": base + "/",
        "model": model,
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "timeout": 180,
        "max_retries": 2,
    }
    # Azure GPT-5 deployments reject temperature=0; use the project convention.
    kwargs["temperature"] = 1.0 if "gpt-5" in model.lower() else 0.0
    if "gpt-5" in model.lower():
        kwargs["model_kwargs"] = {"reasoning_effort": "minimal"}
    return ChatOpenAI(**kwargs)


SYSTEM_PROMPT = """You are auditing thesis-driven research for confirmation bias.

You will receive an initial thesis, the final thesis, and evidence items that
were selected for the final article. Label each evidence item relative to both
theses.

Definitions:
- supports: strengthens the thesis as written.
- qualifies: adds an important caveat, boundary condition, exception, or scope
  limit while leaving the main thesis partly intact.
- refutes: directly contradicts a central claim or empirical premise.
- mixed: contains both supporting and qualifying/refuting content.
- unrelated: does not materially bear on the thesis.

Materiality:
- material: would reasonably affect the article's central argument, thesis
  framing, or a major caveat.
- minor: background detail or low-impact nuance.

Validity:
- valid: the evidence text itself is coherent and usable.
- invalid: the evidence text reports a retrieval/query failure, obvious mismatch,
  or self-undermining result.
- unclear: not enough information to judge.

Important:
- Be conservative. Do not call evidence refuting unless it contradicts a central
  claim, not merely a detail.
- Compare to the INITIAL thesis even if the final thesis later absorbed the
  caveat. This is the main test for early-hypothesis lock-in.
- Return exactly one judgment for every evidence_id provided.
- Return only valid JSON with this shape:
  {
    "judgments": [
      {
        "evidence_id": 1,
        "validity": "valid|invalid|unclear",
        "materiality": "material|minor",
        "stance_vs_initial": "supports|qualifies|refutes|mixed|unrelated",
        "stance_vs_final": "supports|qualifies|refutes|mixed|unrelated",
        "counterevidence_type": "scope_limit|exception|direct_contradiction|alternative_mechanism|measurement_caveat|none",
        "rationale": "...",
        "quoted_basis": "..."
      }
    ]
  }
"""


def user_prompt(run: dict[str, Any], items: list[dict[str, Any]]) -> str:
    prompt_items = [
        {
            **item,
            "question": sanitize_for_policy(item["question"]),
            "finding": sanitize_for_policy(item["finding"]),
        }
        for item in items
    ]
    payload = {
        "topic_id": run["topic_id"],
        "topic": sanitize_for_policy(run["topic"]),
        "initial_thesis": sanitize_for_policy(run["initial_thesis"]),
        "initial_research_strategy": sanitize_for_policy(
            clip(run["initial_research_strategy"], 700)
        ),
        "final_thesis": sanitize_for_policy(clip(run["final_thesis"], 1600)),
        "evidence_items": prompt_items,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_json_response(content: Any) -> TopicJudgments:
    if isinstance(content, list):
        text = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        text = str(content)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        payload = json.loads(text[start : end + 1])
    return TopicJudgments.model_validate(payload)


async def judge_run(
    run: dict[str, Any],
    llm: ChatOpenAI,
    out_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    out_path = out_dir / f"topic_{run['topic_id']}_judgments.json"
    if out_path.exists() and not force:
        return load_json(out_path)

    items = evidence_payload(run)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt(run, items)),
    ]
    response = await llm.ainvoke(messages)
    result = parse_json_response(response.content)
    result_by_id = {j.evidence_id: j.model_dump() for j in result.judgments}

    missing = [item["evidence_id"] for item in items if item["evidence_id"] not in result_by_id]
    if missing:
        raise RuntimeError(f"Topic {run['topic_id']} missing judgments for {missing}")

    output = {
        "topic_id": run["topic_id"],
        "run_dir": str(run["run_dir"].relative_to(PROJECT_ROOT)),
        "topic": run["topic"],
        "initial_thesis_depth": run["initial_thesis_depth"],
        "initial_thesis": run["initial_thesis"],
        "final_thesis": run["final_thesis"],
        "judgments": [
            {
                **item,
                **result_by_id[item["evidence_id"]],
            }
            for item in items
        ],
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    return output


def is_counter_label(label: str) -> bool:
    return label in {"qualifies", "refutes", "mixed"}


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    by_topic = []
    examples = []

    for result in results:
        topic_counts = Counter()
        valid_material_initial_counter = 0
        valid_material_post_initial_counter = 0
        final_absorbed = 0
        final_still_counter = 0

        for judgment in result["judgments"]:
            post = bool(judgment.get("post_initial_thesis"))
            validity = judgment["validity"]
            materiality = judgment["materiality"]
            initial = judgment["stance_vs_initial"]
            final = judgment["stance_vs_final"]

            totals["included_total"] += 1
            totals[f"included_initial_{initial}"] += 1
            totals[f"included_final_{final}"] += 1
            topic_counts[f"initial_{initial}"] += 1
            topic_counts[f"final_{final}"] += 1
            if post:
                totals["included_post_initial_total"] += 1
                totals[f"post_initial_{initial}"] += 1
                topic_counts[f"post_initial_{initial}"] += 1

            countable = validity == "valid" and materiality == "material"
            if countable:
                totals["valid_material_total"] += 1
                if post:
                    totals["valid_material_post_initial_total"] += 1

            if countable and is_counter_label(initial):
                valid_material_initial_counter += 1
                totals["valid_material_initial_counter_total"] += 1
                totals[f"valid_material_initial_counter_{initial}"] += 1
                if post:
                    valid_material_post_initial_counter += 1
                    totals["valid_material_post_initial_counter_total"] += 1
                    totals[f"valid_material_post_initial_counter_{initial}"] += 1
                if final in {"supports", "qualifies", "mixed"}:
                    final_absorbed += 1
                    totals["counter_absorbed_by_final_total"] += 1
                if is_counter_label(final):
                    final_still_counter += 1
                    totals["counter_still_counter_final_total"] += 1
                examples.append(
                    {
                        "topic_id": result["topic_id"],
                        "evidence_id": judgment["evidence_id"],
                        "depth": judgment["depth"],
                        "post_initial_thesis": post,
                        "stance_vs_initial": initial,
                        "stance_vs_final": final,
                        "counterevidence_type": judgment["counterevidence_type"],
                        "rationale": judgment["rationale"],
                        "quoted_basis": judgment["quoted_basis"],
                        "question": judgment["question"],
                    }
                )

        topic_lock_in_failure = (
            valid_material_post_initial_counter > 0 and final_absorbed == 0
        )
        by_topic.append(
            {
                "topic_id": result["topic_id"],
                "included_total": len(result["judgments"]),
                "valid_material_initial_counter": valid_material_initial_counter,
                "valid_material_post_initial_counter": valid_material_post_initial_counter,
                "counter_absorbed_by_final": final_absorbed,
                "counter_still_counter_final": final_still_counter,
                "lock_in_failure": topic_lock_in_failure,
                "counts": dict(topic_counts),
            }
        )

    def ratio(num: int, den: int) -> float | None:
        return None if den == 0 else num / den

    summary = dict(totals)
    summary["counterevidence_inclusion_rate_all_valid_material"] = ratio(
        totals["valid_material_initial_counter_total"], totals["valid_material_total"]
    )
    summary["counterevidence_inclusion_rate_post_initial_valid_material"] = ratio(
        totals["valid_material_post_initial_counter_total"],
        totals["valid_material_post_initial_total"],
    )
    summary["final_absorption_rate_for_initial_counterevidence"] = ratio(
        totals["counter_absorbed_by_final_total"],
        totals["valid_material_initial_counter_total"],
    )
    topics_with_post_counter = sum(
        1 for item in by_topic if item["valid_material_post_initial_counter"] > 0
    )
    topics_lock_in_failure = sum(1 for item in by_topic if item["lock_in_failure"])
    summary["topics_total"] = len(by_topic)
    summary["topics_with_post_initial_counterevidence"] = topics_with_post_counter
    summary["topic_counterevidence_coverage_rate"] = ratio(
        topics_with_post_counter, len(by_topic)
    )
    summary["topics_lock_in_failure"] = topics_lock_in_failure
    summary["topic_lock_in_failure_rate"] = ratio(
        topics_lock_in_failure, topics_with_post_counter
    )

    examples.sort(
        key=lambda x: (
            0 if x["stance_vs_initial"] == "refutes" else 1,
            0 if x["post_initial_thesis"] else 1,
            x["topic_id"],
            x["evidence_id"],
        )
    )
    return {
        "summary": summary,
        "by_topic": sorted(by_topic, key=lambda x: x["topic_id"]),
        "counterevidence_examples": examples[:40],
    }


async def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(root)
    if args.topics:
        allowed = {int(t) if str(t).isdigit() else t for t in args.topics}
        runs = [run for run in runs if run["topic_id"] in allowed]
    runs = sorted(runs, key=lambda run: run["topic_id"])

    if args.extract_only:
        payload = [
            {
                "topic_id": run["topic_id"],
                "run_dir": str(run["run_dir"].relative_to(root)),
                "included_evidence_count": len(evidence_payload(run)),
                "post_initial_evidence_count": sum(
                    1 for item in evidence_payload(run) if item["post_initial_thesis"]
                ),
                "initial_thesis": run["initial_thesis"],
                "final_thesis": clip(run["final_thesis"], 600),
            }
            for run in runs
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    llm = make_llm(args.model)
    results = []
    for index, run in enumerate(runs, start=1):
        print(
            f"[audit] judging topic {run['topic_id']} "
            f"({index}/{len(runs)}; {len(evidence_payload(run))} evidence items)",
            flush=True,
        )
        last_error = None
        for attempt in range(1, args.retries + 1):
            try:
                result = await judge_run(run, llm, out_dir, force=args.force)
                results.append(result)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == args.retries:
                    raise
                await asyncio.sleep(2 * attempt)
        if last_error and len(results) < index:
            raise RuntimeError(f"Topic {run['topic_id']} failed: {last_error}")

    metrics = compute_metrics(results)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    jsonl_path = out_dir / "included_evidence_judgments.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for result in results:
            for judgment in result["judgments"]:
                f.write(
                    json.dumps(
                        {
                            "topic_id": result["topic_id"],
                            "run_dir": result["run_dir"],
                            **judgment,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(json.dumps(metrics["summary"], indent=2, ensure_ascii=False))
    print(f"[audit] wrote {metrics_path.relative_to(root)}")
    print(f"[audit] wrote {jsonl_path.relative_to(root)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--out-dir", default="counterevidence_audit/outputs")
    parser.add_argument("--model", default=os.getenv("COUNTEREVIDENCE_MODEL", "gpt-5"))
    parser.add_argument("--topics", nargs="*", default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only print extracted run/evidence counts; do not call the LLM.",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
