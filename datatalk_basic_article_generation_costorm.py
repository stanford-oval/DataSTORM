from knowledge_storm.utils import load_api_key
load_api_key(toml_file_path='secrets.toml')

import argparse
import json
import os
from dataclasses import dataclass as _dataclass
from datetime import datetime
from typing import List, Optional

from knowledge_storm import DataStormRunnerArguments, DataStormRunner, DataStormLMConfigs
from knowledge_storm.lm import OpenAIModel, AzureOpenAIModel
from knowledge_storm.rm import (
    DatatalkRM, DecompositionAgentRM,
    YouRM, BingSearch, BraveRM, SerperRM, DuckDuckGoSearchRM, TavilySearchRM, SearXNG,
)
from knowledge_storm.collaborative_storm.engine import CollaborativeStormLMConfigs, RunnerArgument, CoStormRunner
from knowledge_storm.datastorm.modules.knowledge_curation import TreeSimulator
from knowledge_storm.collaborative_storm.modules.callback import LocalConsolePrintCallBackHandler
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.langfuse_llm import get_llm, call_llm_with_structured_output
from final_report_gen_utils import (
    generate_report as generate_staged_report,
    generate_report_from_data as generate_staged_report_from_data,
    SourceRegistry, EvidenceRecord,
)


@_dataclass
class WarmstartEvidenceItem:
    question: str
    summary: str
    url: str
    cited_infos: list  # raw Information objects


def extract_warmstart_evidence(conversation_history) -> List["WarmstartEvidenceItem"]:
    items = []
    last_question = ""
    for conv_turn in (conversation_history or []):
        if "question" in (conv_turn.utterance_type or "").lower():
            last_question = conv_turn.utterance or conv_turn.raw_utterance or ""
            continue
        summary = conv_turn.utterance or conv_turn.raw_utterance or ""
        if not summary:
            continue
        question = last_question
        last_question = ""
        urls = []
        cited_infos = []
        if conv_turn.cited_info:
            cited_items = conv_turn.cited_info.items() if isinstance(conv_turn.cited_info, dict) else enumerate(conv_turn.cited_info)
            for _, info in cited_items:
                url = getattr(info, 'url', '') or ''
                if url:
                    urls.append(url)
                cited_infos.append(info)
        items.append(WarmstartEvidenceItem(
            question=question,
            summary=summary,
            url=urls[0] if urls else "",
            cited_infos=cited_infos,
        ))
    return items

# ----------------------------
# Output directory helpers
# ----------------------------
def _ensure_output_dir(output_dir: str) -> str:
    """
    Ensure output_dir exists.

    Note: We keep the root output_dir stable (e.g. "datatalk/"). We append a timestamp suffix to
    the per-topic subdirectory name.
    """
    output_dir = os.path.normpath(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _write_run_metadata(output_dir: str, *, metadata: dict) -> None:
    """
    Write a small JSON file next to artifacts so runs are reproducible.
    Intended to be called after we know the final timestamped output directory.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "run_metadata.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        # Best-effort only; never fail a run just because metadata couldn't be written.
        print(f"[metadata] failed to write run_metadata.json: {e}")


import asyncio
import platform
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class ArticleImpactAnalysis(BaseModel):
    reader_impact: str
    defensibility_assessment: str
    challenging_questions: List[str]


class RevisedArticle(BaseModel):
    article: str


# OpenAI configuration
openai_kwargs = {
    'api_key': os.getenv("AZURE_OPENAI_API_KEY"),
    'temperature': 1.0,
    'top_p': 0.9,
    # Ensure trailing slash: openai SDK module-level base_url path (used by
    # dspy-wrapped OpenAIModel) does *not* normalize; without a trailing slash
    # the URL ends up as ".../v1chat/completions" and 404s.
    'api_base': os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/") + "/"
}
# gpt-4o is no longer served by the genie proxy ("no healthy deployments for
# this model"); gpt-5 matches the datastorm_main_model default. OpenAIModel
# strips the params gpt-5 rejects, so top_p above is ignored for this model.
gpt_4 = OpenAIModel(model='gpt-5', max_tokens=3000, **openai_kwargs)

def init_co_storm(topic: str,
                  retriever: str,
                  warmstart_max_num_experts: int = 3,
                  warmstart_max_turn_per_experts: int=2,
                  node_expansion_trigger_count: int=10,
                  serper_query_params: Optional[dict] = None):
    lm_config: CollaborativeStormLMConfigs = CollaborativeStormLMConfigs()
    openai_kwargs = {
        "api_key": os.getenv("DATASTORM_LITELLM_API_KEY"),
        "api_provider": "openai",
        "temperature": 1.0,
        "top_p": 0.9,
        "api_base": None,
    } if os.getenv('OPENAI_API_TYPE') == 'openai' else {
        "api_key": os.getenv("AZURE_API_KEY"),
        "temperature": 1.0,
        "top_p": 0.9,
        "api_base": os.getenv("AZURE_API_BASE"),
        "api_version": os.getenv("AZURE_API_VERSION"),
    }
    ModelClass = OpenAIModel if os.getenv('OPENAI_API_TYPE') == 'openai' else AzureOpenAIModel
    # If you are using Azure service, make sure the model name matches your own deployed model name.
    gpt_4o_model_name = 'gpt-5'
    if os.getenv('OPENAI_API_TYPE') == 'azure':
        openai_kwargs['api_base'] = os.getenv('AZURE_API_BASE')
        openai_kwargs['api_version'] = os.getenv('AZURE_API_VERSION')

    question_answering_lm = ModelClass(model=gpt_4o_model_name, max_tokens=1000, **openai_kwargs)
    discourse_manage_lm = ModelClass(model=gpt_4o_model_name, max_tokens=500, **openai_kwargs)
    utterance_polishing_lm = ModelClass(model=gpt_4o_model_name, max_tokens=2000, **openai_kwargs)
    warmstart_outline_gen_lm = ModelClass(model=gpt_4o_model_name, max_tokens=500, **openai_kwargs)
    question_asking_lm = ModelClass(model=gpt_4o_model_name, max_tokens=300, **openai_kwargs)
    knowledge_base_lm = ModelClass(model=gpt_4o_model_name, max_tokens=1000, **openai_kwargs)

    lm_config.set_question_answering_lm(question_answering_lm)
    lm_config.set_discourse_manage_lm(discourse_manage_lm)
    lm_config.set_utterance_polishing_lm(utterance_polishing_lm)
    lm_config.set_warmstart_outline_gen_lm(warmstart_outline_gen_lm)
    lm_config.set_question_asking_lm(question_asking_lm)
    lm_config.set_knowledge_base_lm(knowledge_base_lm)

    runner_argument = RunnerArgument(
        topic=topic,
        warmstart_max_num_experts=warmstart_max_num_experts,
        warmstart_max_turn_per_experts=warmstart_max_turn_per_experts,
        node_expansion_trigger_count=node_expansion_trigger_count)
    logging_wrapper = LoggingWrapper(lm_config)
    callback_handler = LocalConsolePrintCallBackHandler()
    match retriever:
        case 'bing':
            rm = BingSearch(bing_search_api=os.getenv('BING_SEARCH_API_KEY'), k=runner_argument.retrieve_top_k)
        case 'you':
             rm = YouRM(ydc_api_key=os.getenv('YDC_API_KEY'), k=runner_argument.retrieve_top_k)
        case 'brave':
            rm = BraveRM(brave_search_api_key=os.getenv('BRAVE_API_KEY'), k=runner_argument.retrieve_top_k)
        case 'duckduckgo':
            rm = DuckDuckGoSearchRM(k=runner_argument.retrieve_top_k, safe_search='On', region='us-en')
        case 'serper':
            default_query_params = {'autocorrect': True, 'num': 10, 'page': 1}
            if serper_query_params:
                default_query_params.update(serper_query_params)
            rm = SerperRM(serper_search_api_key=os.getenv('SERPER_API_KEY'), query_params=default_query_params)
        case 'tavily':
            rm = TavilySearchRM(tavily_search_api_key=os.getenv('TAVILY_API_KEY'), k=runner_argument.retrieve_top_k, include_raw_content=True)
        case 'searxng':
            rm = SearXNG(searxng_api_key=os.getenv('SEARXNG_API_KEY'), k=runner_argument.retrieve_top_k)
        case _:
             raise ValueError(f'Invalid retriever: {retriever}. Choose either "bing", "you", "brave", "duckduckgo", "serper", "tavily", or "searxng"')

    costorm_runner = CoStormRunner(lm_config=lm_config,
                                   runner_argument=runner_argument,
                                   logging_wrapper=logging_wrapper,
                                   rm=rm,
                                   callback_handler=callback_handler)

    return costorm_runner

def init_storm_runner(
    output_dir,
    questions,
    max_tree_depth,
    domain,
    each_level_population_control_num,
    max_global_insights: int = 30,
    db_description=None,
    expansion_max_questions=5,
    enable_followups=True,
    generate_graphs=True,
    consolidate_insights=False,
    disable_upload_to_azure: bool = False,
    enable_python: bool = False,
    langfuse_readonly: bool = False,
    use_decomposition_agent_rm: bool = False,
    article_dir_suffix: str = "",
    internet_rm=None,
    serper_search_params: Optional[dict] = None,
    thesis_generation_depth: int = 3,
    thesis_refinement_interval: int = 2,
    skip_thesis: bool = False,
    include_summary_stats: bool = True,
    datatalk_engine: str = None,
    datastorm_main_model: str = "gpt-5",
):
    """
    Initialize the STORM Wiki runner with database-aware configuration
    
    Args:
        output_dir: Directory for saving outputs
        questions: Initial questions to seed the search
        max_tree_depth: Maximum depth for the search tree
        domain: Domain identifier for the Datatalk RM
        each_level_population_control_num: Number of nodes to retain per level
        max_global_insights: Maximum number of insights retained in global_insights during tree expansion
        db_description: Description of the database being queried
        expansion_max_questions: Maximum number of questions to generate for node expansion
        enable_followups: If True, enable follow-up questions to clean one level
        generate_graphs: If True, generate graphs
        consolidate_insights: If True, enable final consolidation of insights
        disable_upload_to_azure: If True, do not upload SQL results to Azure
        enable_python: If True, allow Datatalk RM to execute Python during retrieval
        langfuse_readonly: If True, reuse Langfuse prompts without storing traces
        use_decomposition_agent_rm: If True, use DecompositionAgentRM instead of DatatalkRM (same init params)
        include_summary_stats: If True, include summary statistics in Datatalk responses
        datatalk_engine: Engine/model to use for Datatalk API calls (e.g., 'gpt-5')
        datastorm_main_model: Model name for internal datastorm LLM calls (summarization, reranking, thesis generation)
    """
    engine_args = DataStormRunnerArguments(
        output_dir=output_dir,
        article_dir_suffix=article_dir_suffix,
        first_level_questions=questions,
        max_tree_depth=max_tree_depth,
        each_level_population_control_num=each_level_population_control_num,
        max_global_insights=max_global_insights,
        db_description=db_description,  # Add database description to arguments
        expansion_max_questions=expansion_max_questions,
        enable_followups=enable_followups,
        generate_graphs=generate_graphs,
        consolidate_insights=consolidate_insights,
        langfuse_readonly=langfuse_readonly,
        thesis_generation_depth=thesis_generation_depth,
        thesis_refinement_interval=thesis_refinement_interval,
        skip_thesis=skip_thesis,
        datastorm_main_model=datastorm_main_model,
    )

    # Set up LM configurations
    lm_configs = DataStormLMConfigs()
    lm_configs.set_conv_simulator_lm(gpt_4)
    lm_configs.set_question_asker_lm(gpt_4)
    
    # Initialize retrieval module
    if use_decomposition_agent_rm:
        rm = DecompositionAgentRM(
            domain=domain,
            disable_upload_to_azure=disable_upload_to_azure,
            enable_python=enable_python,
            langfuse_readonly=langfuse_readonly,
            include_summary_stats=include_summary_stats,
            engine=datatalk_engine,
        )
    else:
        rm = DatatalkRM(
            domain=domain,
            disable_upload_to_azure=disable_upload_to_azure,
            enable_python=enable_python,
            langfuse_readonly=langfuse_readonly,
            include_summary_stats=include_summary_stats,
            engine=datatalk_engine,
        )
    
    # Auto-create internet RM for web search routing if not provided
    if internet_rm is None and os.getenv('SERPER_API_KEY'):
        from knowledge_storm.rm import SerperRM
        internet_rm = SerperRM(query_params=serper_search_params or {})

    # Create runner instance
    runner = DataStormRunner(engine_args, lm_configs, rm, internet_rm=internet_rm)
    
    # Set database description in knowledge curation module
    if hasattr(runner, 'storm_knowledge_curation_module'):
        runner.storm_knowledge_curation_module.database_description = db_description
    
    return runner


### Step 2: Generate questions based on the STORM article
async def generate_questions_from_storm_article(
    topic,
    article,
    db_description,
    num_questions = 3,
    langfuse_readonly: bool = False,
):
    llm = get_llm(model_name="gpt-5")
    class QuestionGenerationResponse(BaseModel):
        questions: List[str]
    
    output = await call_llm_with_structured_output(
        "starting_questions",
        {
            "topic": topic,
            "db_description": db_description,
            "num_questions": num_questions,
            "article": article,
        },
        output_class=QuestionGenerationResponse,
        llm=llm,
        langfuse_readonly=langfuse_readonly,
    )
    output = output.questions
    return output

def load_dataset_descriptions(base_path=None):
    base_path = base_path or os.getenv("INSIGHT_BENCH_NOTEBOOKS", "")
    """
    Dynamically load dataset descriptions from flag-{i}.json files.
    
    Args:
        base_path: Base path where the JSON files are located
        
    Returns:
        Dictionary mapping dataset keys to their descriptions
    """
    descriptions = {}
    i = 1
    while True:
        try:
            file_path = os.path.join(base_path, f"flag-{i}.json")
            if not os.path.exists(file_path):
                break
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                if "metadata" in data and "dataset_description" in data["metadata"]:
                    dataset_key = f"insight_bench/insight_bench_{i}"
                    descriptions[dataset_key] = data["metadata"]["dataset_description"]
            i += 1
        except Exception as e:
            print(f"Error loading dataset {i}: {e}")
            break
    
    return descriptions

db_description_mapping = {
    "acled": "You have access to an ACLED database. Armed Conflict Location & Event Data (ACLED) is a non-profit organization specializing in disaggregated conflict data collection, analysis, and crisis mapping. ACLED codes the dates, actors, locations, fatalities, and types of all reported political violence and demonstration events around the world in real time. We have data up to and until end of 2024.",
    "fec": "You have access to an FEC database storing campaign finance data.",
    "insight_bench/insight_bench_1": "The dataset comprises 500 entries simulating ServiceNow incidents table, detailing various attributes such as category, state, open and close dates, involved personnel, and incident specifics like location, description, and priority. It captures incident management activities with fields like 'opened_at', 'closed_at', 'assigned_to', 'short_description', and 'priority', reflecting the operational handling and urgency of issues across different locations and categories.",
    "sf_311": "You have access to a 311 database storing service requests from the City of San Francisco. The database contains information about various service requests, including the type of request, the location of the request, the status of the request, and the date of the request."
}

# Update the db_description_mapping with dynamically loaded descriptions
db_description_mapping.update(load_dataset_descriptions())




async def regenerate_report_from_dir(
    run_dir: str,
    langfuse_readonly: bool = False,
    model: str = "gpt-5",
) -> str:
    """Regenerate report from an existing run directory using the staged pipeline.

    Always writes to a new timestamped subdirectory (<topic_slug>__YYYYMMDD_HHMMSS)
    inside the source run's output_root_dir, leaving the source run untouched.
    Returns the path to the written co_storm_report.txt.
    """
    import functools
    import shutil

    run_dir = os.path.normpath(os.path.abspath(run_dir))

    # Read source metadata to get topic and output root dir.
    src_metadata_path = os.path.join(run_dir, "run_metadata.json")
    src_metadata = {}
    if os.path.exists(src_metadata_path):
        with open(src_metadata_path, "r", encoding="utf-8") as f:
            src_metadata = json.load(f)
    topic = src_metadata.get("topic", "regenerated")
    output_root_dir = src_metadata.get("output_root_dir") or os.path.dirname(run_dir)

    # Create new timestamped sibling directory using the same naming convention as the
    # live pipeline: base truncated to (125 - len(suffix)) chars + "__YYYYMMDD_HHMMSS".
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"__{timestamp}"
    base_max_len = max(1, 125 - len(suffix))
    slug = topic.replace(" ", "_").replace("/", "_")[:base_max_len]
    work_dir = os.path.join(output_root_dir, f"{slug}{suffix}")
    os.makedirs(work_dir, exist_ok=True)

    # Copy source artifacts needed by the pipeline.
    for fname in ("tree.json", "warmstart_conversation.json", "run_metadata.json", "url_to_info.json", "input.txt"):
        src = os.path.join(run_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(work_dir, fname))

    # Write updated run_metadata capturing regeneration context.
    regen_metadata = dict(src_metadata)
    regen_metadata.update({
        "output_dir": work_dir,
        "regenerated_from": run_dir,
        "generation_module_model": model,
        "regenerated_at": datetime.now().isoformat(),
    })
    _write_run_metadata(work_dir, metadata=regen_metadata)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        functools.partial(generate_staged_report, work_dir, langfuse_readonly=langfuse_readonly, model=model),
    )
    report_content = (result or {}).get("report_content") or ""
    if not report_content:
        raise RuntimeError(f"Staged report generation returned no content for {work_dir}")
    out_path = os.path.join(work_dir, "co_storm_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"  Wrote regenerated report to {out_path}")
    return out_path


async def run_article_critique_phase(
    article: str,
    topic: str,
    dir_tree: str,
    domain: str,
    db_description: str,
    langfuse_readonly: bool,
    max_questions: int = 5,
    disable_upload_to_azure: bool = False,
) -> dict:
    """Run a post-generation critique phase on the article.

    Steps:
      A) LLM assesses reader impact, defensibility, and generates challenging questions.
      B) Evidence gathering via tree search seeded with the challenging questions.
      C) LLM synthesizes findings into a structured critique.
      D) Returns the full structured result dict (caller persists to disk).
    """
    llm = get_llm()

    # --- Step A: Impact & defensibility analysis ---
    impact_analysis: ArticleImpactAnalysis = await call_llm_with_structured_output(
        "article_critique_impact_analysis",
        {"article": article, "max_questions": max_questions},
        output_class=ArticleImpactAnalysis,
        llm=llm,
        context_desc="article critique impact analysis",
        langfuse_readonly=langfuse_readonly,
    )
    challenging_questions = impact_analysis.challenging_questions[:max_questions]
    print(f"  [critique] Impact analysis complete; {len(challenging_questions)} challenging questions generated")

    # --- Step B: Evidence gathering via tree search ---
    search_results = []
    if challenging_questions:
        critique_dir = os.path.join(dir_tree, "critique")
        os.makedirs(critique_dir, exist_ok=True)
        try:
            critique_runner = init_storm_runner(
                output_dir=critique_dir,
                questions=challenging_questions,
                max_tree_depth=1,
                domain=domain,
                each_level_population_control_num=3,
                max_global_insights=15,
                db_description=db_description,
                expansion_max_questions=3,
                enable_followups=False,
                generate_graphs=False,
                consolidate_insights=False,
                langfuse_readonly=langfuse_readonly,
                skip_thesis=True,
                # Without this the critique RM defaults to uploading SQL results
                # to Azure; with an expired SAS token every upload fails, and the
                # raise in DatatalkRM._build_url discards the already-computed
                # answer, silently dropping every critique question.
                disable_upload_to_azure=disable_upload_to_azure,
            )

            _tree_sim = critique_runner.storm_knowledge_curation_module.conv_simulator
            _tree_sim.warmstart_context = (
                f"ORIGINAL ARTICLE:\n{article}\n\n"
                f"DEFENSIBILITY ASSESSMENT:\n{impact_analysis.defensibility_assessment}"
            )

            _, _, critique_tree_json, critique_dir_out, _ = critique_runner.run(
                topic=f"Critique: {topic}",
            )

            # Collect search results per question from the critique tree
            for q in challenging_questions:
                web_findings = []
                db_findings = []
                for node_id in getattr(_tree_sim, 'global_insights', {}):
                    node = _tree_sim.root.get_node_by_id(node_id)
                    if node is None:
                        continue
                    dlg_turn = node.dlg_turn
                    node_question = dlg_turn.user_utterance or ""
                    if q.lower()[:30] not in node_question.lower() and node_question.lower()[:30] not in q.lower():
                        continue
                    summary = dlg_turn.summary or dlg_turn.agent_utterance or ""
                    if dlg_turn.search_results:
                        sr = dlg_turn.search_results[0]
                        src_type = sr.meta.get("source_type", "database") if sr.meta else "database"
                        if src_type == "database":
                            db_findings.append(summary)
                        else:
                            web_findings.append(summary)
                    else:
                        db_findings.append(summary)
                search_results.append({
                    "question": q,
                    "web_findings": web_findings,
                    "db_findings": db_findings,
                })
            print(f"  [critique] Evidence gathering complete; {len(search_results)} question blocks")
        except Exception as e:
            import traceback
            print(f"  [critique] Evidence gathering failed: {e}\n{traceback.format_exc()}")
            search_results = [{"question": q, "web_findings": [], "db_findings": []} for q in challenging_questions]

    # --- Step C: Re-synthesize revised article ---
    evidence_summary_text = "\n\n".join(
        f"Q: {sr['question']}\n"
        f"Web: {'; '.join(sr['web_findings']) or 'none'}\n"
        f"DB: {'; '.join(sr['db_findings']) or 'none'}"
        for sr in search_results
    ) if search_results else "No evidence gathered."

    revised: RevisedArticle = await call_llm_with_structured_output(
        "article_critique_resynthesis",
        {
            "article": article,
            "defensibility_assessment": impact_analysis.defensibility_assessment,
            "evidence_summary": evidence_summary_text,
        },
        output_class=RevisedArticle,
        llm=llm,
        context_desc="article critique resynthesis",
        langfuse_readonly=langfuse_readonly,
    )
    print(f"  [critique] Resynthesis complete")

    return {
        "reader_impact": impact_analysis.reader_impact,
        "defensibility_assessment": impact_analysis.defensibility_assessment,
        "challenging_questions": challenging_questions,
        "search_results": search_results,
        "revised_article": revised.article,
    }


async def run_critique_from_dir(run_dir: str, langfuse_readonly: bool = False) -> str:
    """Run the article critique phase on an existing run directory.

    Reads co_storm_report.txt and run_metadata.json from disk,
    runs the full critique phase, and writes article_critique.json.
    Returns the path to the written file.
    """
    with open(os.path.join(run_dir, "co_storm_report.txt"), "r", encoding="utf-8") as f:
        article = f.read()
    with open(os.path.join(run_dir, "run_metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    topic = meta["topic"]
    domain = meta.get("domain", "acled")
    db_description = meta.get("db_description") or db_description_mapping.get(domain, "")

    critique_result = await run_article_critique_phase(
        article=article,
        topic=topic,
        dir_tree=run_dir,
        domain=domain,
        db_description=db_description,
        langfuse_readonly=langfuse_readonly,
        # Recorded by the original run; without it the critique RM would try to
        # upload and drop every question when the upload fails.
        disable_upload_to_azure=meta.get("disable_upload_to_azure", False),
    )
    out_path = os.path.join(run_dir, "article_critique.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in critique_result.items() if k != "revised_article"}, f, indent=2, ensure_ascii=False)
    revised_article = critique_result.get("revised_article") or ""
    if revised_article:
        revised_path = os.path.join(run_dir, "co_storm_report_revised.txt")
        with open(revised_path, "w", encoding="utf-8") as f:
            f.write(revised_article)
        print(f"  [critique] Revised article written to {revised_path}")
    print(f"  [critique] Written to {out_path}")
    return out_path


async def datastorm_after_warmstart(
    topic,
    costorm_runner,
    domain,
    first_level_questions,
    output_dir,
    max_tree_depth,
    each_level_population_control_num,
    max_global_insights: int = 30,
    db_description = None,
    empty_start = False,
    expansion_max_questions = 5,
    skip_final_article = False,
    enable_followups = True,
    generate_graphs = True,
    consolidate_insights = False,
    disable_upload_to_azure: bool = False,
    enable_python: bool = False,
    langfuse_readonly: bool = False,
    use_topic_as_starting_question: bool = False,
    use_decomposition_agent_rm: bool = False,
    article_dir_suffix: str = "",
    skip_thesis: bool = False,
    include_summary_stats: bool = True,
    datatalk_engine: str = None,
    datastorm_main_model: str = "gpt-5",
):
    if not db_description:
        if domain in db_description_mapping:
            db_description = db_description_mapping[domain]
        else:
            db_description = "You have access to a relevant database. The database description is not provided."
    
    
    print("🟣🟣🟣🟣 Running STEP 2: Generating questions from first STORM article")
    if use_topic_as_starting_question:
        questions = [topic]
    else:
        questions = await generate_questions_from_storm_article(
            topic, 
            article=costorm_runner.knowledge_base.to_report() if not empty_start else None,
            db_description=db_description,
            num_questions=first_level_questions,
            langfuse_readonly=langfuse_readonly,
        )
    assert(type(questions) == list)

    print("🟣🟣🟣🟣 Running STEP 3: Generating tree search retrieval process with Datatalk RM")
    runner = init_storm_runner(
        output_dir=output_dir,
        article_dir_suffix=article_dir_suffix,
        questions=questions, 
        max_tree_depth=max_tree_depth,
        domain=domain,
        each_level_population_control_num=each_level_population_control_num,
        max_global_insights=max_global_insights,
        db_description=db_description,
        expansion_max_questions=expansion_max_questions,
        enable_followups=enable_followups,
        generate_graphs=generate_graphs,
        consolidate_insights=consolidate_insights,
        disable_upload_to_azure=disable_upload_to_azure,
        enable_python=enable_python,
        langfuse_readonly=langfuse_readonly,
        use_decomposition_agent_rm=use_decomposition_agent_rm,
        skip_thesis=skip_thesis,
        include_summary_stats=include_summary_stats,
        datatalk_engine=datatalk_engine,
        datastorm_main_model=datastorm_main_model,
    )

    # Inject warmstart internet evidence as context for thesis generation
    _warmstart_items = extract_warmstart_evidence(getattr(costorm_runner, 'conversation_history', []))
    _tree_sim = runner.storm_knowledge_curation_module.conv_simulator
    if isinstance(_tree_sim, TreeSimulator) and _warmstart_items:
        _tree_sim.warmstart_context = "\n\n".join(
            f"Q: {item.question}\nA: {item.summary}" for item in _warmstart_items
        )

    # topic = 'Pennsylvania campaign donations from in-state v. out-of-state'
    # topic = 'Recent Isreal-Palestine conflicts'
    # TODO: is there a better way to save the tree_json?
    _, additional_evidence_tree, tree_json, dir_tree, _ = runner.run(topic=topic)
    # runner.post_run()
    # runner.summary()

    with open(os.path.join(dir_tree,  "tree.json"), 'w', encoding='utf-8') as f:
        json.dump(tree_json, f, indent=2, ensure_ascii=False)

    # Persist warm-start conversation history so the report can be regenerated later
    # without needing the live costorm_runner object.
    warmstart_history = [
        turn.to_dict()
        for turn in (getattr(costorm_runner, 'conversation_history', None) or [])
        if (turn.utterance or turn.raw_utterance)  # only turns with actual content
    ]
    if warmstart_history:
        with open(os.path.join(dir_tree, "warmstart_conversation.json"), "w", encoding="utf-8") as f:
            json.dump(warmstart_history, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(warmstart_history)} warm-start turns to warmstart_conversation.json")

    # Skip the final article generation if requested
    if skip_final_article:
        print("Skipping final article generation as requested")
        return dir_tree
    
    print("🟣🟣🟣🟣 Running STEP 4: Generating new article using direct LLM call")

    # Assemble evidence from ALL sources into staged_registry + staged_evidence_records.
    # evidence_blocks / citation_map / evidence_text are derived from these after both loops.
    # 1) Co-STORM warm start internet evidence (Step 1)
    # 2) Tree search selected evidence (Step 3) — both database and internet nodes
    staged_registry = SourceRegistry()
    staged_evidence_records = []
    all_search_results = []  # track all Information objects for url_to_info.json

    # --- Warm start internet evidence from Co-STORM (Step 1) ---
    for item in extract_warmstart_evidence(getattr(costorm_runner, 'conversation_history', [])):
        evidence_id = len(staged_evidence_records) + 1
        for info in item.cited_infos:
            all_search_results.append((evidence_id, info))
        citation_id = staged_registry.register(
            source_type="web_search", url=item.url, title=item.question,
            description=item.summary, snippets=[], origin="warmstart",
            meta={"question": item.question},
        )
        staged_evidence_records.append(EvidenceRecord(
            evidence_id=evidence_id,
            citation_id=citation_id, source_type="web_search",
            question=item.question, finding=item.summary, url=item.url,
            node_id="", depth=0,
        ))

    warmstart_count = len(staged_evidence_records)
    if warmstart_count > 0:
        print(f"  Included {warmstart_count} evidence blocks from Co-STORM warm start")

    # --- Tree search selected evidence (Step 3) ---
    _tree_sim = runner.storm_knowledge_curation_module.conv_simulator
    for node_id in getattr(_tree_sim, 'global_insights', {}):
        node = _tree_sim.root.get_node_by_id(node_id)
        if node is None:
            continue
        dlg_turn = node.dlg_turn
        summary = dlg_turn.summary or dlg_turn.agent_utterance or ""
        question = dlg_turn.user_utterance or ""
        url = ""
        source_type = "database"
        title = ""
        description = ""
        snippets = []
        if dlg_turn.search_results:
            sr = dlg_turn.search_results[0]
            url = sr.url or ""
            source_type = sr.meta.get("source_type", "database")
            title = getattr(sr, 'title', '') or ''
            description = getattr(sr, 'description', '') or ''
            snippets = getattr(sr, 'snippets', []) or []
        if source_type == "database" and node.parent:
            parent_question = node.parent.dlg_turn.user_utterance or ""
            if parent_question and question != parent_question:
                question = f"{parent_question}\n{question}"
        evidence_id = len(staged_evidence_records) + 1
        if dlg_turn.search_results:
            all_search_results.append((evidence_id, dlg_turn.search_results[0]))
        citation_id = staged_registry.register(
            source_type=source_type, url=url, title=title,
            description=description, snippets=snippets, origin="tree",
            meta={"node_id": node_id, "depth": node.depth, "question": question},
        )
        staged_evidence_records.append(EvidenceRecord(
            evidence_id=evidence_id,
            citation_id=citation_id, source_type=source_type,
            question=question, finding=summary, url=url,
            node_id=node_id, depth=node.depth,
        ))

    print(f"  Total evidence: {len(staged_evidence_records)} ({warmstart_count} warm start + {len(staged_evidence_records) - warmstart_count} tree search)")

    # Linearize registry into evidence_blocks / citation_map / evidence_text for the critique step
    evidence_blocks = [
        f"[{e.evidence_id}] ({e.source_type.upper()}) Question: {e.question}\nFinding: {e.finding}"
        for e in staged_evidence_records
    ]
    citation_map = {e.evidence_id: e.url for e in staged_evidence_records}
    evidence_text = "\n\n".join(evidence_blocks)

    # Run staged report generation pipeline as the primary final report
    import functools
    loop = asyncio.get_event_loop()
    try:
        staged_result = await loop.run_in_executor(
            None,
            functools.partial(
                generate_staged_report_from_data,
                topic=topic,
                thesis=getattr(runner.storm_knowledge_curation_module.conv_simulator, 'current_thesis', '') or '',
                tree=tree_json,
                warmstart_data=warmstart_history,
                run_dir=dir_tree,
                db_description=db_description,
                langfuse_readonly=langfuse_readonly,
                evidence_records=staged_evidence_records if staged_evidence_records else None,
                registry=staged_registry if staged_evidence_records else None,
                allow_empty_thesis=skip_thesis,
            ),
        )
    except Exception as e:
        import traceback
        print(f"[staged report] Failed: {e}\n{traceback.format_exc()}")
        staged_result = None

    # Get thesis and research_strategy from the conv_simulator
    conv_simulator = runner.storm_knowledge_curation_module.conv_simulator
    thesis = getattr(conv_simulator, 'current_thesis', '') or ''
    research_strategy = getattr(conv_simulator, 'current_research_strategy', '') or ''

    # Get db_description from conv_simulator if not already set
    if not db_description:
        db_description = getattr(conv_simulator, 'db_description', '') or ''

    full_report = (staged_result or {}).get("report_content") or ""
    if not full_report:
        # Fall back to the on-disk staged report only if it was actually written.
        # If staged generation failed (staged_result is None), that file does not
        # exist; surface a clear error instead of a confusing FileNotFoundError.
        staged_md_path = os.path.join(dir_tree, "co_storm_report_staged.md")
        if os.path.exists(staged_md_path):
            print("[staged report] No in-memory report content; falling back to disk read.")
            with open(staged_md_path, "r") as f:
                full_report = f.read()
        else:
            raise RuntimeError(
                "Staged report generation produced no output and no "
                "co_storm_report_staged.md was written — see the '[staged report] "
                "Failed: ...' message above for the underlying cause."
            )

    fact_check_stats = (staged_result or {}).get("fact_check_stats") or {}
    if fact_check_stats:
        print(f"  [fact_check] {fact_check_stats.get('issues_found', 0)}/{fact_check_stats.get('total_checked', 0)} issue(s) found across {len(fact_check_stats.get('per_section', []))} sections")

    with open(os.path.join(dir_tree, "co_storm_report.txt"), "w") as f:
        f.write(full_report)

    print("🔍 Running STEP 6: Article critique phase")
    try:
        critique_result = await run_article_critique_phase(
            article=full_report,
            topic=topic,
            dir_tree=dir_tree,
            domain=domain,
            db_description=db_description or "",
            langfuse_readonly=langfuse_readonly,
            disable_upload_to_azure=disable_upload_to_azure,
        )
        with open(os.path.join(dir_tree, "article_critique.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in critique_result.items() if k != "revised_article"}, f, indent=2, ensure_ascii=False)
        revised_article = critique_result.get("revised_article") or ""
        if revised_article:
            with open(os.path.join(dir_tree, "co_storm_report_revised.txt"), "w", encoding="utf-8") as f:
                f.write(revised_article)
        print(f"  [critique] {len(critique_result.get('challenging_questions', []))} questions explored")
    except Exception as e:
        import traceback
        print(f"[critique] Failed: {e}\n{traceback.format_exc()}")

    # Build url_to_info.json from all evidence sources (warm start + tree search)
    url_to_info_data = {"url_to_unified_index": {}, "url_to_info": {}}
    for citation_idx, info in all_search_results:
        url = getattr(info, 'url', '') or ''
        if url and url not in url_to_info_data["url_to_info"]:
            url_to_info_data["url_to_unified_index"][url] = citation_idx
            info_dict = info.to_dict() if hasattr(info, 'to_dict') else {"url": url}
            info_dict["citation_uuid"] = citation_idx
            url_to_info_data["url_to_info"][url] = info_dict

    with open(os.path.join(dir_tree, "url_to_info.json"), "w", encoding="utf-8") as f:
        json.dump(url_to_info_data, f, indent=2, ensure_ascii=False)

    return dir_tree

async def process_generation(
    topic,
    output_dir,
    first_level_questions=3,
    max_tree_depth=1,
    warmstart_max_num_experts=3,
    warmstart_max_turn_per_experts=2,
    node_expansion_trigger_count=10,
    costorm_retriever="serper",
    domain="acled",
    each_level_population_control_num=5,
    max_global_insights=30,
    db_description=None,
    no_warm_start=False,
    expansion_max_questions=5,
    skip_final_article=False,
    enable_followups=True,
    generate_graphs=True,
    consolidate_insights=False,
    disable_upload_to_azure: bool = False,
    enable_python: bool = False,
    langfuse_readonly: bool = False,
    use_topic_as_starting_question: bool = False,
    use_decomposition_agent_rm: bool = False,
    append_timestamp_to_output_dir: bool = True,
    serper_query_params: Optional[dict] = None,
    skip_thesis: bool = False,
    generation_module_model: str = "gpt-5",
    include_summary_stats: bool = True,
    datatalk_engine: str = None,
    datastorm_main_model: str = "gpt-5",
):
    output_dir = _ensure_output_dir(output_dir)
    article_dir_suffix = ""
    # Always append timestamp to the per-topic folder name, e.g.
    # datatalk/<topic_slug>__YYYYMMDD_HHMMSS
    #
    if append_timestamp_to_output_dir:
        article_dir_suffix = "__" + datetime.now().strftime("%Y%m%d_%H%M%S")
    print("🟣🟣🟣🟣 Running STEP 1: Generating first STORM article")
    costorm_runner = init_co_storm(
        topic=topic,
        retriever=costorm_retriever,
        warmstart_max_num_experts=warmstart_max_num_experts,
        warmstart_max_turn_per_experts=warmstart_max_turn_per_experts,
        node_expansion_trigger_count=node_expansion_trigger_count,
        serper_query_params=serper_query_params
    )
    
    empty_start = False
    if not no_warm_start:
        costorm_runner.warm_start()
        empty_start = True
    
    run_output_dir = await datastorm_after_warmstart(
        topic,
        costorm_runner,
        domain=domain,
        first_level_questions=first_level_questions,
        output_dir=output_dir,
        max_tree_depth=max_tree_depth,
        each_level_population_control_num=each_level_population_control_num,
        max_global_insights=max_global_insights,
        db_description=db_description,
        empty_start=empty_start,
        expansion_max_questions=expansion_max_questions,
        skip_final_article=skip_final_article,
        enable_followups=enable_followups,
        generate_graphs=generate_graphs,
        consolidate_insights=consolidate_insights,
        disable_upload_to_azure=disable_upload_to_azure,
        enable_python=enable_python,
        langfuse_readonly=langfuse_readonly,
        use_topic_as_starting_question=use_topic_as_starting_question,
        use_decomposition_agent_rm=use_decomposition_agent_rm,
        article_dir_suffix=article_dir_suffix,
        skip_thesis=skip_thesis,
        include_summary_stats=include_summary_stats,
        datatalk_engine=datatalk_engine,
        datastorm_main_model=datastorm_main_model,
    )
    
    # Record run metadata inside the final artifact directory.
    _write_run_metadata(
        run_output_dir,
        metadata={
            "topic": topic,
            "output_root_dir": output_dir,
            "output_dir": run_output_dir,
            "domain": domain,
            "serper_query_params": serper_query_params,
            "first_level_questions": first_level_questions,
            "max_tree_depth": max_tree_depth,
            "warmstart_max_num_experts": warmstart_max_num_experts,
            "warmstart_max_turn_per_experts": warmstart_max_turn_per_experts,
            "node_expansion_trigger_count": node_expansion_trigger_count,
            "costorm_retriever": costorm_retriever,
            "each_level_population_control_num": each_level_population_control_num,
            "max_global_insights": max_global_insights,
            "db_description": db_description,
            "no_warm_start": no_warm_start,
            "expansion_max_questions": expansion_max_questions,
            "skip_final_article": skip_final_article,
            "enable_followups": enable_followups,
            "generate_graphs": generate_graphs,
            "consolidate_insights": consolidate_insights,
            "disable_upload_to_azure": disable_upload_to_azure,
            "enable_python": enable_python,
            "langfuse_readonly": langfuse_readonly,
            "use_topic_as_starting_question": use_topic_as_starting_question,
            "use_decomposition_agent_rm": use_decomposition_agent_rm,
            "append_timestamp_to_output_dir": append_timestamp_to_output_dir,
            "generation_module_model": generation_module_model,
            "include_summary_stats": include_summary_stats,
            "datatalk_engine": datatalk_engine,
            "datastorm_main_model": datastorm_main_model,
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "timestamp": datetime.now().isoformat(),
        },
    )

    # Backwards-compatible response:
    # - output_dir: resolved, timestamped per-topic output directory (the one that contains artifacts)
    # - output_root_dir: the stable root output directory passed in / ensured on disk
    return {"status": "completed", "output_dir": run_output_dir, "output_root_dir": output_dir}

if __name__ == "__main__":
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Datatalk Basic Article Generation")
        parser.add_argument('--output_dir', type=str, help='Directory to save the output')
        parser.add_argument('--first_level_questions', type=int, default=3, help='Number of questions for the first level')
        parser.add_argument('--max_tree_depth', type=int, default=1, help='Maximum tree depth')
        parser.add_argument(
            '--warmstart_max_num_experts',
            type=int,
            default=3,
            help='Max number of experts in perspective-guided QA during warm start.'
        )
        parser.add_argument(
            '--warmstart_max_turn_per_experts',
            type=int,
            default=2,
            help='Max number of turns per perspective during warm start.'
        )
        parser.add_argument(
            '--node_expansion_trigger_count',
            type=int,
            default=10,
            help='Trigger node expansion for nodes that contain more than N snippets.'
        )
        parser.add_argument('--costorm_retriever', type=str, choices=['bing', 'you', 'brave', 'serper', 'duckduckgo', 'tavily', 'searxng'],
                            help='The search engine API to use for retrieving information.', default="serper")
        parser.add_argument('--topic', type=str, help='Topic for the article generation')
        parser.add_argument('--domain', type=str, help='Domain for the Datatalk RM', default="acled")
        parser.add_argument(
            '--db_description',
            type=str,
            help='Description of the database to inform Datatalk RM (overrides domain mapping)'
        )
        parser.add_argument(
            '--each_level_population_control_num',
            type=int,
            default=5,
            help='In Datatalk RM, on each level, how many nodes to retain'
        )
        parser.add_argument(
            '--max_global_insights',
            type=int,
            default=30,
            help='Maximum number of insights retained in global_insights during tree expansion'
        )
        parser.add_argument(
            '--no_warm_start',
            action='store_true',
            help='Flag to indicate whether to perform warm start'
        )
        parser.add_argument(
            '--api_mode',
            action='store_true',
            help='Start a FastAPI server to handle requests through an API'
        )
        parser.add_argument(
            '--port',
            type=int,
            default=8000,
            help='Port to run the API server on (only used with --api_mode)'
        )
        parser.add_argument(
            '--expansion_max_questions',
            type=int,
            default=5,
            help='Maximum number of questions to generate for node expansion'
        )
        parser.add_argument(
            '--skip_final_article',
            action='store_true',
            help='Skip step 4 (generating the final article using tree search results)'
        )
        parser.add_argument(
            '--disable_followups',
            action='store_true',
            help='Disable follow-up questions (enabled by default)'
        )
        parser.add_argument(
            '--disable_graphs',
            action='store_true',
            help='Disable graph generation'
        )
        parser.add_argument(
            '--enable_consolidate_insights',
            action='store_true',
            help='Enable final consolidation of insights (disabled by default)'
        )
        parser.add_argument(
            '--disable_upload_to_azure',
            action='store_true',
            help='Disable uploading SQL result artifacts to Azure'
        )
        parser.add_argument(
            '--enable_python',
            action='store_true',
            help='Enable Python execution for Datatalk RM queries'
        )
        parser.add_argument(
            '--langfuse_readonly',
            action='store_true',
            help='Load prompts from Langfuse without recording new traces'
        )
        parser.add_argument(
            '--use_topic_as_starting_question',
            action='store_true',
            help='Use the provided topic directly as the starting question instead of generating questions with the LLM'
        )
        parser.add_argument(
            '--use_decomposition_agent_rm',
            action='store_true',
            help='If set, use DecompositionAgentRM instead of DatatalkRM (same init params)'
        )
        parser.add_argument(
            '--skip_thesis',
            action='store_true',
            help='Skip thesis generation and refinement; global insights still run and drive question generation'
        )
        parser.add_argument(
            '--no_summary_stats',
            action='store_true',
            help='Disable bottom-up inductive summary statistics on Datatalk RM responses (default: enabled)'
        )
        parser.add_argument(
            '--datatalk_engine',
            type=str,
            default=None,
            help='Engine/model name to use for Datatalk RM calls (e.g. gpt-5).'
        )
        parser.add_argument(
            '--datastorm_main_model',
            type=str,
            default='gpt-5',
            help='Model name for internal DataSTORM LLM calls (planner, thesis, etc.).'
        )
        parser.add_argument(
            '--serper_query_params',
            type=str,
            help='JSON string of Serper query parameters (e.g., \'{"tbs": "cdr:1,cd_max:01/02/2025"}\')'
        )
        parser.add_argument(
            '--regenerate',
            type=str,
            metavar='RUN_DIR',
            help='Regenerate co_storm_report.txt from an existing run directory (reads tree.json + run_metadata.json)'
        )
        parser.add_argument(
            '--generation_module_model',
            type=str,
            default='gpt-5',
            help='Model to use for the article generation module (default: gpt-5).'
        )
        parser.add_argument(
            '--regenerate_staged',
            type=str,
            metavar='RUN_DIR',
            help='Generate staged report from an existing run directory using the multi-step pipeline'
        )
        parser.add_argument(
            '--critique',
            type=str,
            metavar='RUN_DIR',
            help='Run article critique phase on an existing run directory (reads co_storm_report.txt + run_metadata.json)'
        )
        return parser.parse_args()

    # Define FastAPI request model
    class ArticleGenerationRequest(BaseModel):
        output_dir: str
        topic: str
        domain: str = "acled"
        db_description: Optional[str] = None
        first_level_questions: int = 3
        max_tree_depth: int = 1
        warmstart_max_num_experts: int = 3
        warmstart_max_turn_per_experts: int = 2
        node_expansion_trigger_count: int = 10
        costorm_retriever: str = "serper"
        each_level_population_control_num: int = 5
        max_global_insights: int = 30
        no_warm_start: bool = False
        expansion_max_questions: int = 5
        skip_final_article: bool = False
        enable_followups: bool = True
        generate_graphs: bool = True
        consolidate_insights: bool = False
        disable_upload_to_azure: bool = False
        enable_python: bool = False
        langfuse_readonly: bool = False
        use_topic_as_starting_question: bool = False
        use_decomposition_agent_rm: bool = False
        append_timestamp_to_output_dir: bool = True
        serper_query_params: Optional[dict] = None
        skip_thesis: bool = False
        include_summary_stats: bool = True
        datatalk_engine: Optional[str] = None
        datastorm_main_model: str = "gpt-5"

    # Define basic LLM configs
    args = parse_arguments()

    if args.api_mode:
        app = FastAPI()

        @app.post("/generate_article")
        async def generate_article(request: ArticleGenerationRequest):
            # try:
            result = await process_generation(
                topic=request.topic,
                output_dir=request.output_dir,
                first_level_questions=request.first_level_questions,
                max_tree_depth=request.max_tree_depth,
                warmstart_max_num_experts=request.warmstart_max_num_experts,
                warmstart_max_turn_per_experts=request.warmstart_max_turn_per_experts,
                node_expansion_trigger_count=request.node_expansion_trigger_count,
                costorm_retriever=request.costorm_retriever,
                domain=request.domain,
                each_level_population_control_num=request.each_level_population_control_num,
                max_global_insights=request.max_global_insights,
                db_description=request.db_description,
                no_warm_start=request.no_warm_start,
                expansion_max_questions=request.expansion_max_questions,
                skip_final_article=request.skip_final_article,
                enable_followups=request.enable_followups,
                generate_graphs=request.generate_graphs,
                consolidate_insights=request.consolidate_insights,
                disable_upload_to_azure=request.disable_upload_to_azure,
                enable_python=request.enable_python,
                langfuse_readonly=request.langfuse_readonly,
                use_topic_as_starting_question=request.use_topic_as_starting_question,
                use_decomposition_agent_rm=request.use_decomposition_agent_rm,
                append_timestamp_to_output_dir=request.append_timestamp_to_output_dir,
                serper_query_params=request.serper_query_params,
                skip_thesis=request.skip_thesis,
                include_summary_stats=request.include_summary_stats,
                datatalk_engine=request.datatalk_engine,
                datastorm_main_model=request.datastorm_main_model,
            )
            return result
            # except Exception as e:
                # raise HTTPException(status_code=500, detail=str(e))

        @app.post("/regenerate_report")
        async def api_regenerate_report(request: dict):
            run_dir = request.get("run_dir", "")
            if not run_dir:
                raise HTTPException(status_code=400, detail="run_dir is required")
            langfuse_readonly = request.get("langfuse_readonly", False)
            out_path = await regenerate_report_from_dir(run_dir, langfuse_readonly=langfuse_readonly)
            return {"status": "completed", "output_path": out_path}

        print(f"Starting FastAPI server on port {args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    elif args.regenerate:
        import asyncio
        asyncio.run(regenerate_report_from_dir(
            args.regenerate,
            langfuse_readonly=args.langfuse_readonly,
            model=args.generation_module_model,
        ))
    elif args.regenerate_staged:
        generate_staged_report(run_dir=args.regenerate_staged, langfuse_readonly=args.langfuse_readonly)
    elif args.critique:
        asyncio.run(run_critique_from_dir(args.critique, langfuse_readonly=args.langfuse_readonly))
    else:
        import asyncio
        assert(args.topic is not None)
        assert(args.output_dir is not None)

        # Parse serper_query_params from JSON string if provided
        serper_params = None
        if args.serper_query_params:
            try:
                serper_params = json.loads(args.serper_query_params)
            except json.JSONDecodeError as e:
                print(f"Error parsing serper_query_params JSON: {e}")
                exit(1)

        asyncio.run(process_generation(
            topic=args.topic,
            output_dir=args.output_dir,
            first_level_questions=args.first_level_questions,
            max_tree_depth=args.max_tree_depth,
            warmstart_max_num_experts=args.warmstart_max_num_experts,
            warmstart_max_turn_per_experts=args.warmstart_max_turn_per_experts,
            node_expansion_trigger_count=args.node_expansion_trigger_count,
            costorm_retriever=args.costorm_retriever,
            domain=args.domain,
            db_description=args.db_description,
            each_level_population_control_num=args.each_level_population_control_num,
            max_global_insights=args.max_global_insights,
            no_warm_start=args.no_warm_start,
            expansion_max_questions=args.expansion_max_questions,
            skip_final_article=args.skip_final_article,
            enable_followups=not args.disable_followups,
            generate_graphs=not args.disable_graphs,
            consolidate_insights=args.enable_consolidate_insights,
            disable_upload_to_azure=args.disable_upload_to_azure,
            enable_python=args.enable_python,
            langfuse_readonly=args.langfuse_readonly,
            use_topic_as_starting_question=args.use_topic_as_starting_question,
            use_decomposition_agent_rm=args.use_decomposition_agent_rm,
            serper_query_params=serper_params,
            skip_thesis=args.skip_thesis,
            include_summary_stats=not args.no_summary_stats,
            datatalk_engine=args.datatalk_engine,
            datastorm_main_model=args.datastorm_main_model,
            generation_module_model=args.generation_module_model,
        ))
