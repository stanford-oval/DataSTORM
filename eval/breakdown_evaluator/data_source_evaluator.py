#!/usr/bin/env python3
"""
Data Source Evaluator

This evaluator classifies whether the insights in a generated article are
derived from ACLED data or other sources.

Flow:
1. Break down the generated article into atomic insights using `datastorm_acled_eval_breakdown_prompt`
2. For each insight, classify whether it is sourced from ACLED data
3. Aggregate: count and percentage of ACLED-sourced insights

Usage:
    python data_source_evaluator.py \
        --generated_file /path/to/generated.md \
        --output_file /path/to/output.json \
        --prompt "Topic description"
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
    "data_source": "datastorm_acled_eval_data_source_prompt",
}


class AtomicInsight(BaseModel):
    """A single atomic insight extracted from an article."""
    insight: str = Field(description="A single, atomic insight or fact from the article")
    source_context: str = Field(description="Brief context about where this insight appears in the article")


class InsightBreakdownOutput(BaseModel):
    """Output from breaking down an article into atomic insights."""
    insights: list[AtomicInsight] = Field(description="List of atomic insights extracted from the article")


class DataSourceOutput(BaseModel):
    """Structured output for data source classification."""
    is_acled_data: bool = Field(description="Whether the insight is derived from ACLED data")
    reasoning: str = Field(description="Brief explanation for the classification")


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
    """Break down an article into atomic insights using the Langfuse prompt."""
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


async def classify_data_source(
    article_topic: str,
    evidence: str,
    client: LangfuseLLMClient,
) -> dict:
    """
    Classify whether a single piece of evidence is derived from ACLED data.

    Returns:
        Dict with {evidence, is_acled_data, reasoning}
    """
    variables = {
        "article_topic": article_topic,
        "evidence": evidence,
    }

    try:
        result = await client.generate_structured(
            prompt_id=PROMPT_IDS["data_source"],
            variables=variables,
            output_class=DataSourceOutput,
            context_desc="data source classification",
        )
        return {
            "evidence": evidence,
            "is_acled_data": result.is_acled_data,
            "reasoning": result.reasoning,
        }
    except Exception as e:
        return {
            "evidence": evidence,
            "is_acled_data": None,
            "reasoning": f"Error: {e}",
        }


async def evaluate_data_sources(
    generated_article: str,
    article_topic: str,
    client: LangfuseLLMClient,
) -> dict:
    """
    Classify all insights in a generated article as ACLED-sourced or not.

    Args:
        generated_article: The generated article to evaluate
        article_topic: Topic/prompt describing the article
        client: LangfuseLLMClient instance

    Returns:
        Dict with data source metrics and detailed breakdown
    """
    # Step 1: Break down the generated article into atomic insights
    print("[Data Source] Breaking down generated article into insights...")
    generated_insights = await breakdown_article_to_insights(generated_article, client)
    print(f"[Data Source] Found {len(generated_insights)} insights in generated article")

    if not generated_insights:
        return {
            "total_insights": 0,
            "acled_count": 0,
            "non_acled_count": 0,
            "pct_acled": 0.0,
            "details": [],
            "error": "No insights found in generated article",
        }

    # Step 2: Classify each insight (in parallel)
    print(f"[Data Source] Classifying {len(generated_insights)} insights...")

    tasks = [
        classify_data_source(
            article_topic=article_topic,
            evidence=insight.insight,
            client=client,
        )
        for insight in generated_insights
    ]

    results = await asyncio.gather(*tasks)

    # Step 3: Aggregate
    valid_results = [r for r in results if r["is_acled_data"] is not None]
    acled_count = sum(1 for r in valid_results if r["is_acled_data"])
    non_acled_count = sum(1 for r in valid_results if not r["is_acled_data"])
    total_valid = len(valid_results)
    pct_acled = acled_count / total_valid if total_valid > 0 else 0.0

    # Add context to each result
    for i, r in enumerate(results):
        r["source_context"] = generated_insights[i].source_context

    return {
        "total_insights": len(generated_insights),
        "total_classified": total_valid,
        "acled_count": acled_count,
        "non_acled_count": non_acled_count,
        "pct_acled": pct_acled,
        "details": results,
        "insights": [
            {"insight": ins.insight, "context": ins.source_context}
            for ins in generated_insights
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
    generated_article = read_text_arg(args.generated, args.generated_file)

    if not generated_article:
        raise ValueError("Generated article is required (text or file).")

    article_topic = args.prompt or ""
    client = LangfuseLLMClient(model_name=args.model)

    result = await evaluate_data_sources(
        generated_article=generated_article,
        article_topic=article_topic,
        client=client,
    )

    # Add metadata
    result["prompt"] = article_topic

    # Print summary
    print(f"\n{'='*50}")
    print("Data Source Classification Results")
    print(f"{'='*50}")
    print(f"Total insights: {result['total_insights']}")
    print(f"Total classified: {result['total_classified']}")
    print(f"ACLED-sourced: {result['acled_count']}")
    print(f"Non-ACLED: {result['non_acled_count']}")
    print(f"% ACLED: {result['pct_acled']:.2%}")
    print(f"{'='*50}")

    # Print non-ACLED insights
    non_acled = [d for d in result["details"] if d["is_acled_data"] is False]
    if non_acled:
        print(f"\nNon-ACLED insights ({len(non_acled)} total):")
        for i, item in enumerate(non_acled[:10], 1):
            print(f"  {i}. {item['evidence'][:100]}...")
        if len(non_acled) > 10:
            print(f"  ... and {len(non_acled) - 10} more")

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
        description="Classify insights in a generated article as ACLED-sourced or not."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Article topic / task prompt.",
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
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
