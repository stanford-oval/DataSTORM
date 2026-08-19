import argparse
import asyncio
import json
import logging
from typing import Optional

from pydantic import BaseModel

from deep_research_bench_evaluator.utils.score_calculator import calculate_weighted_scores
from knowledge_storm.langfuse_llm import call_llm_with_structured_output, get_llm

logger = logging.getLogger(__name__)

PROMPT_IDS = {
    "weight": "drb_eval_dimension_weight_en",
    "criteria": {
        "comprehensiveness": "drb_eval_criteria_comprehensiveness_en",
        "insight": "drb_eval_criteria_insight_en",
        "instruction_following": "drb_eval_criteria_instruction_following_en",
        "readability": "drb_eval_criteria_readability_en",
    },
    "merged_score": "drb_eval_merged_score_en",
}

EXPECTED_DIMS = ["comprehensiveness", "insight", "instruction_following", "readability"]


class WeightOutput(BaseModel):
    comprehensiveness: float
    insight: float
    instruction_following: float
    readability: float


class CriterionItem(BaseModel):
    criterion: str
    explanation: str
    weight: float


class CriteriaOutput(BaseModel):
    criteria: list[CriterionItem]


class ScoreItem(BaseModel):
    criterion: str
    analysis: str
    article_1_score: float
    article_2_score: float


class MergedScoreOutput(BaseModel):
    comprehensiveness: list[ScoreItem]
    insight: list[ScoreItem]
    instruction_following: list[ScoreItem]
    readability: list[ScoreItem]


def validate_weights(data, expected_sum=1.0, tolerance=1e-6):
    if isinstance(data, dict):
        if not data:
            return False
        total_weight = sum(float(value) for value in data.values())
        return abs(total_weight - expected_sum) < tolerance
    if isinstance(data, list):
        if not data or not all(isinstance(item, dict) and "weight" in item for item in data):
            return False
        total_weight = sum(float(item["weight"]) for item in data)
        return abs(total_weight - expected_sum) < tolerance
    return False


def normalize_criteria_weights(criteria_list: list[dict], min_sum: float = 0.5, max_sum: float = 1.5) -> list[dict] | None:
    """
    Normalize criteria weights to sum to 1.0 if the current sum is within [min_sum, max_sum].
    Returns None if the sum is outside the acceptable range or input is invalid.
    """
    if not criteria_list or not all(isinstance(item, dict) and "weight" in item for item in criteria_list):
        return None
    
    total_weight = sum(float(item["weight"]) for item in criteria_list)
    
    if total_weight < min_sum or total_weight > max_sum:
        return None
    
    # Normalize weights to sum to 1.0
    normalized = []
    for item in criteria_list:
        new_item = item.copy()
        new_item["weight"] = float(item["weight"]) / total_weight
        normalized.append(new_item)
    
    return normalized


def round_weights_and_adjust(weights, decimal_places=2):
    rounded = {dim: round(float(weight), decimal_places) for dim, weight in weights.items()}
    total = sum(rounded.values())
    diff = 1.0 - total
    if abs(diff) > 1e-10:
        rounded["readability"] = round(rounded["readability"] + diff, decimal_places)
    return rounded


class LangfuseLLMClient:
    def __init__(self, model_name: Optional[str] = None):
        self.llm = get_llm(model_name=model_name or "gpt-5")

    async def generate_structured(self, prompt_id: str, variables: dict, output_class, context_desc: str):
        response = await call_llm_with_structured_output(
            prompt_id=prompt_id,
            variables=variables,
            output_class=output_class,
            llm=self.llm,
            context_desc=context_desc,
        )
        if response is None:
            raise RuntimeError(f"LLM returned no response for {context_desc}")
        return response


async def generate_weights_multiple_times(
    prompt, ai_client, sample_count, retries, retry_delay
):
    weights_samples = []
    weight_prompt_id = PROMPT_IDS["weight"]
    variables = {"task_prompt": prompt}

    for _ in range(sample_count):
        for attempt in range(retries):
            weights_output = await ai_client.generate_structured(
                prompt_id=weight_prompt_id,
                variables=variables,
                output_class=WeightOutput,
                context_desc="dimension weight",
            )
            try:
                parsed_weights = weights_output.model_dump()
                if validate_weights(parsed_weights):
                    weights_samples.append(parsed_weights)
                    break
            except Exception:
                if attempt == retries - 1:
                    raise
            if attempt < retries - 1:
                await asyncio.sleep(retry_delay)

    if not weights_samples:
        return None

    dimensions = set()
    for sample in weights_samples:
        dimensions.update(sample.keys())

    avg_weights = {}
    for dim in dimensions:
        values = [sample.get(dim, 0) for sample in weights_samples if dim in sample]
        if len(values) == len(weights_samples):
            avg_weights[dim] = sum(values) / len(values)

    weight_sum = sum(avg_weights.values())
    for dim in avg_weights:
        avg_weights[dim] = avg_weights[dim] / weight_sum

    return round_weights_and_adjust(avg_weights, decimal_places=2)


async def generate_criteria(prompt, ai_client, retries, retry_delay, sample_count):
    weights = await generate_weights_multiple_times(prompt, ai_client, sample_count, retries, retry_delay)
    if not weights:
        raise RuntimeError("Failed to generate dimension weights.")

    criteria_prompts = PROMPT_IDS["criteria"]
    criterions = {}

    for dim_name, criteria_prompt_id in criteria_prompts.items():
        variables = {"task_prompt": prompt}
        for attempt in range(retries):
            criteria_output = await ai_client.generate_structured(
                prompt_id=criteria_prompt_id,
                variables=variables,
                output_class=CriteriaOutput,
                context_desc=f"criteria {dim_name}",
            )
            try:
                parsed_criteria = [item.model_dump() for item in criteria_output.criteria]
                if parsed_criteria:
                    # Try to normalize weights if they sum to a reasonable range (0.5-1.5)
                    normalized_criteria = normalize_criteria_weights(parsed_criteria)
                    if normalized_criteria:
                        criterions[dim_name] = normalized_criteria
                        break
                    # Fall back to strict validation if normalization fails
                    elif validate_weights(parsed_criteria):
                        criterions[dim_name] = parsed_criteria
                        break
            except Exception:
                if attempt == retries - 1:
                    raise
            if attempt < retries - 1:
                await asyncio.sleep(retry_delay)

        if dim_name not in criterions:
            raise RuntimeError(f"Failed to generate criteria for dimension: {dim_name}")

    return {
        "dimension_weight": weights,
        "criterions": criterions,
    }


def format_criteria_list(criteria_data):
    criteria_for_prompt = {}
    criterions_dict = criteria_data.get("criterions", {})
    for dim, criterions_list in criterions_dict.items():
        if not isinstance(criterions_list, list):
            continue
        criteria_for_prompt[dim] = []
        for item in criterions_list:
            if isinstance(item, dict) and "criterion" in item and "explanation" in item:
                criteria_for_prompt[dim].append(
                    {"criterion": item["criterion"], "explanation": item["explanation"]}
                )
    return json.dumps(criteria_for_prompt, ensure_ascii=False, indent=2)


def read_text_arg(value, file_path):
    if value:
        return value
    if not file_path:
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


async def evaluate(args):
    reference_article = read_text_arg(args.reference, args.reference_file)
    generated_article = read_text_arg(args.generated, args.generated_file)
    if not reference_article or not generated_article:
        raise ValueError("Both reference and generated articles are required (text or file).")

    llm_client = LangfuseLLMClient()

    criteria_data = await generate_criteria(
        args.prompt, llm_client, args.retries, args.retry_delay, args.sample_count
    )

    async def _merged_score(criteria_list_str, desc):
        """Run the merged-score judge call; returns the parsed model or None (no raise)."""
        return await call_llm_with_structured_output(
            prompt_id=PROMPT_IDS["merged_score"],
            variables={
                "task_prompt": args.prompt,
                "article_1": generated_article,
                "article_2": reference_article,
                "criteria_list": criteria_list_str,
            },
            output_class=MergedScoreOutput,
            llm=llm_client.llm,
            context_desc=desc,
        )

    # Fast path: original single merged-score call over all dimensions at once.
    # Behavior is unchanged whenever this succeeds.
    merged = await _merged_score(format_criteria_list(criteria_data), "merged score")

    blocked_dims: list[str] = []
    if merged is not None:
        llm_output = merged.model_dump()
    else:
        # Defensive fallback. The single call came back empty — typically Azure's
        # content filter rejecting a violence-heavy report in one shot. Score one
        # dimension at a time so a single blocked dimension degrades to 0 (and is
        # excluded from both target and reference totals) instead of losing the
        # entire report. We always produce a result rather than raising.
        logger.warning("Merged-score call returned None; falling back to per-dimension scoring.")
        llm_output = {dim: [] for dim in EXPECTED_DIMS}
        fallback_retries = min(max(1, args.retries), 2)
        for dim in EXPECTED_DIMS:
            one_dim_criteria = {
                "dimension_weight": criteria_data.get("dimension_weight", {}),
                "criterions": {dim: criteria_data.get("criterions", {}).get(dim, [])},
            }
            dim_criteria_str = format_criteria_list(one_dim_criteria)
            dim_out = None
            for attempt in range(fallback_retries):
                dim_out = await _merged_score(dim_criteria_str, f"merged score [{dim}]")
                if dim_out is not None:
                    break
                if attempt < fallback_retries - 1:
                    await asyncio.sleep(args.retry_delay)
            if dim_out is None:
                blocked_dims.append(dim)
                logger.warning(
                    "Dimension '%s' could not be scored (content filter / no response); treating as 0.",
                    dim,
                )
                continue
            dim_scores = getattr(dim_out, dim, None) or []
            llm_output[dim] = [item.model_dump() for item in dim_scores]
        if len(blocked_dims) == len(EXPECTED_DIMS):
            logger.error(
                "All %d dimensions were blocked; overall score will be 0 for this report.",
                len(EXPECTED_DIMS),
            )

    scores = calculate_weighted_scores(llm_output, criteria_data, "en")
    target_total = scores["target"]["total"]
    reference_total = scores["reference"]["total"]
    overall_score = 0
    if target_total + reference_total > 0:
        overall_score = target_total / (target_total + reference_total)

    normalized_dims = {}
    for dim in EXPECTED_DIMS:
        dim_key = f"{dim}_weighted_avg"
        target_score = scores["target"]["dims"].get(dim_key, 0)
        reference_score = scores["reference"]["dims"].get(dim_key, 0)
        if target_score + reference_score > 0:
            normalized_dims[dim] = target_score / (target_score + reference_score)
        else:
            normalized_dims[dim] = 0

    result = {
        "prompt": args.prompt,
        "comprehensiveness": normalized_dims.get("comprehensiveness", 0),
        "insight": normalized_dims.get("insight", 0),
        "instruction_following": normalized_dims.get("instruction_following", 0),
        "readability": normalized_dims.get("readability", 0),
        "overall_score": overall_score,
    }
    if blocked_dims:
        # Transparency: these dimensions could not be judged (content filter / no
        # response) and were counted as 0. The overall reflects the remaining ones.
        result["blocked_dims"] = blocked_dims
        result["degraded"] = True

    output_json = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output_json + "\n")
    else:
        print(output_json)


def build_parser():
    parser = argparse.ArgumentParser(description="Run one-off RACE evaluation for a prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="Task prompt text.")
    parser.add_argument("--reference", type=str, default=None, help="Reference article text.")
    parser.add_argument("--reference_file", type=str, default=None, help="Path to reference article.")
    parser.add_argument("--generated", type=str, default=None, help="Generated article text.")
    parser.add_argument("--generated_file", type=str, default=None, help="Path to generated article.")
    parser.add_argument("--language", type=str, choices=["en"], default="en")
    parser.add_argument("--output_file", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--sample_count", type=int, default=5, help="Samples for weight averaging.")
    parser.add_argument("--retries", type=int, default=5, help="LLM retry attempts.")
    parser.add_argument("--retry_delay", type=int, default=5, help="LLM retry delay in seconds.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
