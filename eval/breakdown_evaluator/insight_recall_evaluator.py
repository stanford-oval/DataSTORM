#!/usr/bin/env python3
"""
Insight Recall Evaluator

This evaluator compares two articles by:
1. Breaking down each article into atomic insights using `datastorm_acled_eval_breakdown_prompt`
2. For each insight in the reference, checking if it's matched by any insight in the predicted article
3. Computing recall = matched_reference_insights / total_reference_insights

Usage:
    python insight_recall_evaluator.py \
        --reference_file /path/to/reference.md \
        --generated_file /path/to/generated.md \
        --output_file /path/to/output.json
"""

import argparse
import asyncio
import json
from typing import Optional

from pydantic import BaseModel, Field

from knowledge_storm.langfuse_llm import call_llm_with_structured_output, get_llm


# Prompt IDs in Langfuse
PROMPT_IDS = {
    "breakdown": "datastorm_acled_eval_breakdown_prompt",
    "insight_match": "datastorm_acled_eval_insight_match_prompt",
}


class AtomicInsight(BaseModel):
    """A single atomic insight extracted from an article."""
    insight: str = Field(description="A single, atomic insight or fact from the article")
    source_context: str = Field(description="Brief context about where this insight appears in the article")


class InsightBreakdownOutput(BaseModel):
    """Output from breaking down an article into atomic insights."""
    insights: list[AtomicInsight] = Field(description="List of atomic insights extracted from the article")


class InsightMatchOutput(BaseModel):
    """Output from comparing two insights for semantic equivalence."""
    is_match: bool = Field(description="Whether the two insights are semantically equivalent or the predicted insight captures the reference insight")
    reasoning: str = Field(description="Brief explanation of why the insights match or don't match")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0)


class LangfuseLLMClient:
    """Client for calling LLMs with Langfuse prompt management."""

    def __init__(self, model_name: Optional[str] = None):
        self.llm = get_llm(model_name=model_name or "gpt-5")

    async def generate_structured(
        self,
        prompt_id: str,
        variables: dict,
        output_class,
        context_desc: str,
        use_cache: bool = True,
    ):
        response = await call_llm_with_structured_output(
            prompt_id=prompt_id,
            variables=variables,
            output_class=output_class,
            llm=self.llm,
            context_desc=context_desc,
            use_cache=use_cache,
        )
        if response is None:
            raise RuntimeError(f"LLM returned no response for {context_desc}")
        return response


async def breakdown_article_to_insights(
    article: str,
    client: LangfuseLLMClient,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> list[AtomicInsight]:
    """
    Break down an article into atomic insights using the Langfuse prompt.

    Args:
        article: The article text to break down
        client: LangfuseLLMClient instance
        retries: Number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        List of AtomicInsight objects
    """
    variables = {"article": article}

    for attempt in range(retries):
        try:
            result = await client.generate_structured(
                prompt_id=PROMPT_IDS["breakdown"],
                variables=variables,
                output_class=InsightBreakdownOutput,
                context_desc="article insight breakdown",
            )
            return result.insights
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to breakdown article after {retries} attempts: {e}")
            await asyncio.sleep(retry_delay)

    return []


async def check_insight_match(
    reference_insight: str,
    predicted_insight: str,
    client: LangfuseLLMClient,
    confidence_threshold: float = 0.7,
) -> tuple[bool, str, float]:
    """
    Check if a predicted insight matches a reference insight.

    Args:
        reference_insight: The insight from the reference article
        predicted_insight: The insight from the predicted article
        client: LangfuseLLMClient instance
        confidence_threshold: Minimum confidence to consider a match

    Returns:
        Tuple of (is_match, reasoning, confidence)
    """
    variables = {
        "reference_insight": reference_insight,
        "predicted_insight": predicted_insight,
    }

    try:
        result = await client.generate_structured(
            prompt_id=PROMPT_IDS["insight_match"],
            variables=variables,
            output_class=InsightMatchOutput,
            context_desc="insight match check",
        )
        # Consider it a match if LLM says match AND confidence is above threshold
        is_match = result.is_match and result.confidence >= confidence_threshold
        return is_match, result.reasoning, result.confidence
    except Exception as e:
        # On error, assume no match
        return False, f"Error checking match: {e}", 0.0


async def find_best_match_for_insight(
    reference_insight: AtomicInsight,
    predicted_insights: list[AtomicInsight],
    client: LangfuseLLMClient,
    confidence_threshold: float = 0.7,
) -> dict:
    """
    Find the best matching predicted insight for a reference insight.

    Args:
        reference_insight: The reference insight to match
        predicted_insights: List of predicted insights to search
        client: LangfuseLLMClient instance
        confidence_threshold: Minimum confidence to consider a match

    Returns:
        Dict with match info: {matched, best_match, confidence, reasoning}
    """
    best_match = None
    best_confidence = 0.0
    best_reasoning = ""

    # Check against all predicted insights in parallel
    tasks = [
        check_insight_match(
            reference_insight.insight,
            pred.insight,
            client,
            confidence_threshold,
        )
        for pred in predicted_insights
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            continue
        is_match, reasoning, confidence = result
        if confidence > best_confidence:
            best_confidence = confidence
            best_match = predicted_insights[i].insight if is_match else None
            best_reasoning = reasoning
            if is_match:
                best_match = predicted_insights[i].insight

    return {
        "reference_insight": reference_insight.insight,
        "reference_context": reference_insight.source_context,
        "matched": best_match is not None,
        "best_match": best_match,
        "confidence": best_confidence,
        "reasoning": best_reasoning,
    }


async def calculate_insight_recall(
    reference_article: str,
    predicted_article: str,
    client: LangfuseLLMClient,
    confidence_threshold: float = 0.7,
) -> dict:
    """
    Calculate recall of insights from reference article found in predicted article.

    Args:
        reference_article: The ground truth reference article
        predicted_article: The predicted/generated article
        client: LangfuseLLMClient instance
        confidence_threshold: Minimum confidence to consider a match

    Returns:
        Dict with recall metrics and detailed breakdown
    """
    # Step 1: Break down both articles into atomic insights
    print("[Evaluator] Breaking down reference article into insights...")
    reference_insights = await breakdown_article_to_insights(reference_article, client)
    print(f"[Evaluator] Found {len(reference_insights)} insights in reference article")

    print("[Evaluator] Breaking down predicted article into insights...")
    predicted_insights = await breakdown_article_to_insights(predicted_article, client)
    print(f"[Evaluator] Found {len(predicted_insights)} insights in predicted article")

    if not reference_insights:
        return {
            "recall": 0.0,
            "matched_count": 0,
            "total_reference_insights": 0,
            "total_predicted_insights": len(predicted_insights),
            "details": [],
            "error": "No insights found in reference article",
        }

    # Step 2: For each reference insight, find best match in predicted insights
    print(f"[Evaluator] Matching {len(reference_insights)} reference insights against {len(predicted_insights)} predicted insights...")

    match_tasks = [
        find_best_match_for_insight(ref, predicted_insights, client, confidence_threshold)
        for ref in reference_insights
    ]

    match_results = await asyncio.gather(*match_tasks)

    # Step 3: Calculate recall
    matched_count = sum(1 for r in match_results if r["matched"])
    recall = matched_count / len(reference_insights) if reference_insights else 0.0

    return {
        "recall": recall,
        "matched_count": matched_count,
        "total_reference_insights": len(reference_insights),
        "total_predicted_insights": len(predicted_insights),
        "details": match_results,
        "reference_insights": [
            {"insight": ins.insight, "context": ins.source_context}
            for ins in reference_insights
        ],
        "predicted_insights": [
            {"insight": ins.insight, "context": ins.source_context}
            for ins in predicted_insights
        ],
    }


def read_text_arg(value: Optional[str], file_path: Optional[str]) -> str:
    """Read text from direct value or file path."""
    if value:
        return value
    if not file_path:
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


async def evaluate(args):
    """Main evaluation function."""
    reference_article = read_text_arg(args.reference, args.reference_file)
    generated_article = read_text_arg(args.generated, args.generated_file)

    if not reference_article or not generated_article:
        raise ValueError("Both reference and generated articles are required (text or file).")

    client = LangfuseLLMClient(model_name=args.model)

    result = await calculate_insight_recall(
        reference_article=reference_article,
        predicted_article=generated_article,
        client=client,
        confidence_threshold=args.confidence_threshold,
    )

    # Add metadata
    result["prompt"] = args.prompt if args.prompt else ""
    result["confidence_threshold"] = args.confidence_threshold

    # Print summary
    print(f"\n{'='*50}")
    print("Insight Recall Evaluation Results")
    print(f"{'='*50}")
    print(f"Recall: {result['recall']:.4f} ({result['matched_count']}/{result['total_reference_insights']})")
    print(f"Reference insights: {result['total_reference_insights']}")
    print(f"Predicted insights: {result['total_predicted_insights']}")
    print(f"{'='*50}")

    # Print unmatched insights
    unmatched = [d for d in result["details"] if not d["matched"]]
    if unmatched:
        print(f"\nUnmatched reference insights ({len(unmatched)}):")
        for i, u in enumerate(unmatched[:10], 1):  # Show first 10
            print(f"  {i}. {u['reference_insight'][:100]}...")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")

    # Output to file or stdout
    output_json = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output_json + "\n")
        print(f"\nResults saved to: {args.output_file}")
    else:
        print("\nFull results:")
        print(output_json)

    return result


def build_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate insight recall between reference and generated articles."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional task prompt for context.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference article text (direct).",
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        default=None,
        help="Path to reference article file.",
    )
    parser.add_argument(
        "--generated",
        type=str,
        default=None,
        help="Generated article text (direct).",
    )
    parser.add_argument(
        "--generated_file",
        type=str,
        default=None,
        help="Path to generated article file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="Model to use for evaluation (default: gpt-5).",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for insight matching (default: 0.7).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
