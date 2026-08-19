"""Library entry points for the datatalk SQL agent.

Two callable surfaces are exposed for callers that don't run under chainlit:

  on_chat_start(domain, ...) -> DatatalkParser
      Build a DatatalkParser configured for a given domain. Returns the
      parser; callers thread it back into run_single_message.

  run_single_message(message, conversation_history, semantic_parser_class, ...)
      Execute a single user query through the agent's LangGraph runnable
      and produce the same response dict the FastAPI endpoint returns.

These are extracted from datatalk_domains/backend/agent/chainlit_frontend_2_5.py
so the existing FastAPI endpoint and the upcoming in-process callers
inside storm can use the same code path. All chainlit-only behavior
(URL transcript replay, cl.user_session, cl.Message/Step/File/Image/Action
rendering) is gated behind `enable_chainlit=True`; library callers pass
False (the default) and never trigger any chainlit imports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib

DATATALK_DOMAINS_ROOT = os.getenv(
    "DATATALK_DOMAINS_ROOT",
    str(pathlib.Path(__file__).resolve().parent.parent.parent / "datatalk_domains"),
)
import urllib.parse
from dataclasses import dataclass
from decimal import Decimal
from multiprocessing import Value
from threading import Lock
from typing import Any, Dict, List, Optional

import litellm

from knowledge_storm.langfuse_llm import call_llm_with_structured_output, get_llm
from knowledge_storm.datatalk_agent.agent import (
    DatatalkParser,
    retrieve_relevant_domain_specific_instructions,
)
from knowledge_storm.datatalk_agent.sql_utils import prepare_initialize
from knowledge_storm.datatalk_agent.state import (
    Action,
    SqlQuery,
    json_to_panda_markdown_token_limited,
    locate_last_generated_base64_image,
)
from knowledge_storm.datatalk_agent.utils import postprocess_entities

from langchain.schema.runnable.config import RunnableConfig
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Cost tracking — preserved from chainlit_frontend_2_5.py.
# ---------------------------------------------------------------------------
global_cost_counter = Value("d", 0.0)
counter_lock = Lock()


async def track_cost_callback(kwargs, completion_response, start_time, end_time):
    try:
        response_cost = kwargs["response_cost"]
        with counter_lock:
            global_cost_counter.value += response_cost
    except Exception as e:
        print(f"track_func_token_callback encountered exception: {e}")


# Registers cost tracker on each litellm call across processes & threads.
litellm.success_callback = [track_cost_callback]


# ---------------------------------------------------------------------------
# Response models / helpers reused from the original chainlit module.
# ---------------------------------------------------------------------------
@dataclass
class SqlReporterResponse(BaseModel):
    generated_sql: Optional[str]
    technical_explanation_of_SQL: Optional[str]
    simplified_explanation_of_SQL: Optional[str]
    limitations: Optional[str]
    report: Optional[str]
    suggested_next_turn_natural_language_queries: Optional[List[str]]
    include_images: Optional[bool] = False


class ActionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Action):
            return obj.to_dict()
        return super().default(obj)


def map_step_name_to_natural_names(step_name: str) -> str:
    return {
        "execute_sql": "Building queries",
        "get_tables": "Understand database structure",
        "retrieve_tables_details": "Get table details",
        "entity_linking": "Linking entities",
        "location_linking": "Resolving locations",
        "execute_python_from_sql": "Executing Python code based on SQL results",
    }.get(step_name, step_name)


def check_source(sqls: List[SqlQuery]) -> List[str]:
    res = set()
    fec_tables = (
        "candidate_master_2023_2024",
        "all_candidates_2023_2024",
        "candidate_committee_linkage_2023_2024",
        "current_campaigns_for_house_and_senate_2023_2024",
        "committee_master_2023_2024",
        "pac_and_party_summary_2023_2024",
        "contributions_by_individuals_2023_2024",
        "contributions_from_comm_to_cand_and_ind_expenditures_2023_2024",
        "any_transaction_from_one_committee_to_another_2023_2024",
    )
    for s in (q.sql for q in sqls):
        if "pac_details" in s:
            res.add("[OpenSecrets.org](https://www.OpenSecrets.org)")
        if any(t in s for t in fec_tables):
            res.add("FEC")
        if len(res) >= 2:
            break
    return list(res)


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------
async def on_chat_start(
    domain: Optional[str] = None,
    enable_chainlit: bool = False,
    engine: str = "gpt-5",
    enable_python: bool = False,
    langfuse_readonly: bool = False,
) -> DatatalkParser:
    """Build a DatatalkParser configured for the requested domain.

    Library callers pass `enable_chainlit=False` and use the returned
    parser directly. UI callers pass `enable_chainlit=True`; the chainlit
    user-session bookkeeping and URL-transcript replay are deferred to
    the chainlit-aware wrapper that still lives in chainlit_frontend_2_5.py
    in the datatalk_domains repo.
    """
    chat_profile = domain
    if enable_chainlit:
        # Late import: chainlit instantiates an app at import time and is
        # only valid in a chainlit-served process.
        import chainlit as cl

        if chat_profile is None:
            chat_profile = cl.user_session.get("chat_profile")

    if chat_profile is None:
        raise ValueError("on_chat_start requires `domain` when not running under chainlit")

    # Domain-specific feature flags.
    # ACLED sessions allow Python execution (e.g. post-processing SQL results).
    if chat_profile == "acled":
        enable_python = True

    table_w_ids, table_schema_lst = prepare_initialize(
        f"{DATATALK_DOMAINS_ROOT}/{chat_profile}/_lookup_table.csv"
    )

    # The litellm_proxy/ prefix routes the SUQL compiler's LLM calls through the
    # genie proxy (LITELLM_PROXY_API_BASE/KEY). A bare "azure/gpt-5" instead sends
    # them to LiteLLM's Azure provider, which reads AZURE_API_KEY/AZURE_API_BASE
    # and fails with AuthenticationError; the compiler then silently yields no
    # rows ("temp_table_... does not exist") for any answer()-in-WHERE query.
    suql_model_name = "litellm_proxy/gpt-5"
    # SUQL servers live at these ports — see
    # ~/datatalk_domains/backend/ingestion/ingestion_domains/acled_suql_test.py.
    embedding_server_address = os.getenv("SUQL_EMBEDDING_SERVER", "http://127.0.0.1:8505")
    free_text_server_address = os.getenv("SUQL_FREE_TEXT_SERVER", "http://127.0.0.1:8510")

    db_details_path = f"{DATATALK_DOMAINS_ROOT}/{chat_profile}/db_details.json"
    db_type = "postgres"
    db_secrets_file: Optional[str] = None
    suql_enabled = False
    database_name = chat_profile
    db_path: Optional[str] = None
    if os.path.exists(db_details_path):
        with open(db_details_path, "r") as fd:
            db_details = json.load(fd)
        db_type = db_details.get("db_type", db_type)
        db_secrets_file = db_details.get("db_secrets_file", db_secrets_file)
        suql_enabled = db_details.get("suql_enabled", suql_enabled)
        database_name = db_details.get("database_name", database_name)
        db_path = db_details.get("db_path", db_path)
        # Per-domain override for table_w_ids — useful when SUQL needs an
        # explicit primary-key column that the lookup CSV doesn't carry
        # (e.g. acled events → event_id_cnty).
        table_w_ids_override = db_details.get("table_w_ids")
        if table_w_ids_override:
            table_w_ids = {**table_w_ids, **table_w_ids_override}

    available_actions = ["get_tables", "retrieve_tables_details", "execute_sql"]

    DatatalkParser.initialize(engine=engine)
    semantic_parser_class = DatatalkParser(
        engine=engine,
        table_w_ids=table_w_ids,
        database_name=database_name,
        suql_model_name=suql_model_name,
        embedding_server_address=embedding_server_address,
        free_text_server_address=free_text_server_address,
        table_schema=table_schema_lst,
        domain_specific_instructions=f"{DATATALK_DOMAINS_ROOT}/{chat_profile}/domain_specific_instructions.csv",
        db_type=db_type,
        db_secrets_file=db_secrets_file,
        suql_enabled=suql_enabled,
        enable_python=enable_python,
        langfuse_readonly=langfuse_readonly,
        available_actions=available_actions,
        db_path=db_path,
    )

    if enable_chainlit:
        import chainlit as cl  # already imported above; idempotent
        cl.user_session.set("conversation_history", [])
        cl.user_session.set("turn_id", 0)
        cl.user_session.set("proposed_rule", None)
        cl.user_session.set("proposed_rule_history", [])
        cl.user_session.set("table_w_ids", table_w_ids)
        cl.user_session.set("embedding_server_address", embedding_server_address)
        cl.user_session.set("source_file_mapping", {})
        cl.user_session.set("suql_model_name", suql_model_name)
        cl.user_session.set("db_type", db_type)
        cl.user_session.set("db_secrets_file", db_secrets_file)
        cl.user_session.set("suql_enabled", suql_enabled)
        cl.user_session.set("database_name", database_name)
        cl.user_session.set("db_path", db_path)
        cl.user_session.set("langfuse_readonly", langfuse_readonly)
        cl.user_session.set("semantic_parser_class", semantic_parser_class)

    return semantic_parser_class


async def run_single_message(
    message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    chainlit_msg_object: Any = None,
    semantic_parser_class: Optional[DatatalkParser] = None,
    save_to_local: Optional[str] = None,
    save_result_to_csv: bool = False,
    include_summary_stats: bool = False,
    designation: Optional[str] = None,
    engine: str = "gpt-5",
) -> Dict[str, Any]:
    """Execute a single agent turn and return the response dict.

    `chainlit_msg_object` is the only chainlit-specific argument: pass
    None for library use (no UI rendering). When it is None, no chainlit
    APIs are imported or invoked.
    """
    if conversation_history is None:
        conversation_history = []
    enable_chainlit = chainlit_msg_object is not None

    callbacks: List[Any] = []
    if enable_chainlit:
        import chainlit as cl
        callbacks = [
            cl.AsyncLangchainCallbackHandler(
                stream_final_answer=True,
                to_ignore=[
                    "Runnable", "<lambda>", "initialize_state", "LangGraph",
                    "execute_sql", "get_tables", "retrieve_tables_details",
                    "get_examples", "controller", "__start__", "start", "router",
                    "reporter", "stop", "_write", "verify_domain_specific_instructions",
                    "entity_linking", "location_linking", "execute_python_from_sql",
                ],
            )
        ]

    # Carry forward any entity-linking substitution dict from the previous turn.
    if conversation_history and "entity_linking_results" in conversation_history[-1]:
        entity_linking_results = conversation_history[-1]["entity_linking_results"]
    else:
        entity_linking_results = {}

    if semantic_parser_class is None:
        if enable_chainlit:
            import chainlit as cl
            semantic_parser_class = cl.user_session.get("semantic_parser_class")
        else:
            raise ValueError(
                "run_single_message requires `semantic_parser_class` when not running under chainlit"
            )

    langfuse_readonly = getattr(semantic_parser_class, "langfuse_readonly", False)

    state: Dict[str, Any] = {
        "question": message,
        "conversation_history": conversation_history,
        "generated_sqls": [],
        "actions": [],
        "entity_linking_results": entity_linking_results,
        "langfuse_readonly": langfuse_readonly,
    }

    # Fast-path: caller asked us to execute the message verbatim as SQL.
    if designation == "SQL" and message:
        direct_state = {
            "question": message,
            "conversation_history": conversation_history,
            "actions": [],
            "table_w_ids": getattr(semantic_parser_class, "table_w_ids", {}),
            "database_name": getattr(semantic_parser_class, "database_name", None),
            "embedding_server_address": getattr(semantic_parser_class, "embedding_server_address", None),
            "source_file_mapping": getattr(semantic_parser_class, "source_file_mapping", {}),
            "suql_model_name": getattr(semantic_parser_class, "suql_model_name", None),
            "db_type": getattr(semantic_parser_class, "db_type", "postgres"),
            "db_secrets_file": getattr(semantic_parser_class, "db_secrets_file", None),
            "suql_enabled": getattr(semantic_parser_class, "suql_enabled", False),
            "db_path": getattr(semantic_parser_class, "db_path", None),
            "langfuse_readonly": langfuse_readonly,
        }

        sql_query_object = await DatatalkParser.sql_chain(
            message, direct_state, limit_query=" LIMIT 10000"
        )
        sql_result = sql_query_object.execution_result_sample
        result_count = (
            len(sql_query_object.execution_result_full_dict)
            if sql_query_object.execution_result_full_dict is not None
            else 0
        )

        msg_content = f"```sql\n{message}\n```\n\n## Result\n\n{sql_result}\n\n"
        summary = msg_content

        if include_summary_stats and result_count > 10:
            summary_stats = sql_query_object.get_table_summary_statistics()
            msg_content += "## Summary Statistics\n\n"
            summary += "## Summary Statistics\n\n"
            for column, stats in summary_stats.items():
                msg_content += f"### {column}\n"
                summary += f"### {column}\n"
                for stat, value in stats.items():
                    msg_content += f"{stat}: {value}\n"
                    summary += f"{stat}: {value}\n"

        if result_count >= 10000:
            note = "**Note:** The result set could be beyond 10,000 rows. The final SQL results were limited to 10,000 rows."
            msg_content += note
            summary += note

        if enable_chainlit:
            chainlit_msg_object.content = msg_content

        conversation_history.append(
            {
                "question": message.strip(),
                "action_history": [],
                "entity_linking_results": {},
                "response": msg_content,
            }
        )

        if enable_chainlit:
            import chainlit as cl
            cl.user_session.set("conversation_history", conversation_history)

        res = {
            "agent_response": msg_content,
            "preprocessed_sql": message,
            "generated_sql": message,
            "conversation_history": json.dumps(conversation_history, cls=ActionEncoder),
            "result_count": result_count,
            "summary": summary,
            "sql_result": sql_result,
        }

        return _maybe_save_to_local(
            res, sql_query_object, save_to_local, save_result_to_csv
        )

    # Standard path: run the agent's LangGraph runnable.
    async for chunk in semantic_parser_class.runnable.with_config(
        {"recursion_limit": 60, "max_concurrency": 50}
    ).astream_events(
        {
            "question": message,
            "conversation_history": conversation_history,
            "entity_linking_results": entity_linking_results,
        },
        config=RunnableConfig(
            callbacks=callbacks,
            tags=[semantic_parser_class.database_name],
        ),
        version="v1",
    ):
        if (
            chunk["name"] in [
                "execute_sql", "get_tables", "retrieve_tables_details",
                "entity_linking", "location_linking",
                "execute_python_from_sql",
            ]
            and "data" in chunk
            and "input" in chunk["data"]
            and "actions" in chunk["data"]["input"]
            and chunk["data"]["input"]["actions"]
            and chunk["event"] == "on_chain_end"
            and chunk["tags"][0].startswith("graph:")
            and "langsmith:hidden" not in chunk["tags"]
        ):
            action = chunk["data"]["input"]["actions"][-1]
            if enable_chainlit:
                step_name = map_step_name_to_natural_names(action.action_name)
                # Step counter is 1-based across the run.
                state.setdefault("_step_counter", 1)
                await action.print_chainlit(
                    f"Step {state['_step_counter']}: {step_name}"
                )
                state["_step_counter"] += 1
            state = chunk["data"]["input"]

    # Build the structured reporter call from the actions accumulated in `state`.
    action_history: List[str] = []
    domain_specific_instructions: set = set()
    if "actions" in state:
        actions = state["actions"]
        get_tables_schema_results: List[Any] = []
        for i, a in enumerate(actions):
            include_observation = True
            if i < len(actions) - 7 and a.action_name in ["execute_sql"]:
                include_observation = False
            elif a.action_name == "stop":
                include_observation = False
            if a.action_name == "get_tables_schema":
                if a.observation in get_tables_schema_results:
                    include_observation = False
                else:
                    get_tables_schema_results.append(a.observation)
            action_history.append(a.to_jinja_string(include_observation))

        for a in reversed(actions):
            if a.action_name == "execute_sql":
                domain_specific_instructions = domain_specific_instructions.union(
                    set(
                        retrieve_relevant_domain_specific_instructions(
                            a.action_argument,
                            state["domain_specific_instructions"],
                            controller_reporter="reporter",
                        )
                    )
                )

    result = await call_llm_with_structured_output(
        "datatalk_reporter",
        {
            "question": state["question"],
            "conversation_history": state["conversation_history"],
            "action_history": action_history,
            "domain_specific_instructions": list(domain_specific_instructions),
        },
        SqlReporterResponse,
        get_llm(model_name=engine),
        image_dicts=locate_last_generated_base64_image(
            state["actions"], return_llm_compatible_dict=True
        ),
        langfuse_readonly=state.get("langfuse_readonly", langfuse_readonly),
    )

    report = result.report
    preprocessed_sql = result.generated_sql
    postprocessed_sql = result.generated_sql
    technical_explanation_of_SQL = result.technical_explanation_of_SQL
    simplified_explanation_of_SQL = result.simplified_explanation_of_SQL
    limitations = result.limitations
    suggested_next_turn_queries = result.suggested_next_turn_natural_language_queries
    include_images = result.include_images

    msg_content = ""
    summary = ""
    result_count = 0
    sql_result = None
    sql_query_object = None
    if preprocessed_sql:
        substitution_dict = state["entity_linking_results"]
        postprocessed_sql = await postprocess_entities(postprocessed_sql, substitution_dict)

        sql_query_object = await DatatalkParser.sql_chain(
            postprocessed_sql, state, limit_query=" LIMIT 10000"
        )
        if sql_query_object.execution_result_full_dict is not None:
            sql_result = json_to_panda_markdown_token_limited(
                sql_query_object.execution_result_full_dict
            )
        else:
            sql_result = sql_query_object.execution_result_sample

        msg_content = "I attempted to answer your question by the following query\n\n"
        msg_content += f"```sql\n{postprocessed_sql}\n```\n\n"
        summary = f"```sql\n{preprocessed_sql}\n```\n\n"
        msg_content += f"### Simplified Explanation of SQL\n\n{simplified_explanation_of_SQL}\n\n"
        summary += f"### Simplified Explanation of SQL\n\n{simplified_explanation_of_SQL}\n\n"
        msg_content += f"### Technical Explanation of SQL\n\n{technical_explanation_of_SQL}\n\n"
        msg_content += f"### Limitations\n\n{limitations}\n\n"
        msg_content += f"## Result\n\n{sql_result}\n\n"
        summary += f"## Result\n\n{sql_result}\n\n"
        msg_content += f"## Report\n\n{report}\n\n"
        summary += f"## Report\n\n{report}\n\n"

        if sql_query_object.execution_result_full_dict is not None:
            result_count = len(sql_query_object.execution_result_full_dict)
            if include_summary_stats and result_count > 10:
                summary_stats = sql_query_object.get_table_summary_statistics()
                msg_content += "## Summary Statistics\n\n"
                summary += "## Summary Statistics\n\n"
                for column, stats in summary_stats.items():
                    msg_content += f"### {column}\n"
                    summary += f"### {column}\n"
                    for stat, value in stats.items():
                        msg_content += f"{stat}: {value}\n"
                        summary += f"{stat}: {value}\n"

            if result_count >= 10000:
                note = "**Note:** The result set could be beyond 10,000 rows. The final SQL results were limited to 10,000 rows."
                msg_content += note
                summary += note

    if enable_chainlit:
        import chainlit as cl
        chainlit_msg_object.content = msg_content
        chainlit_msg_object.elements = []
        if postprocessed_sql:
            chainlit_msg_object.elements.append(
                cl.File(
                    name="View Full Results, Edit, or Share",
                    url=os.getenv("DATATALK_SQL_CONSOLE_URL", "")
                    + urllib.parse.quote(postprocessed_sql),
                )
            )
        if include_images:
            import base64
            for i, image_dict in enumerate(
                locate_last_generated_base64_image(state["actions"])
            ):
                img_bytes = base64.b64decode(image_dict)
                chainlit_msg_object.elements.append(
                    cl.Image(
                        content=img_bytes,
                        mime="image/png",
                        name=f"plot_{i+1}.png",
                        display="inline",
                    )
                )
        await chainlit_msg_object.send()

    conversation_history.append(
        {
            "question": state["question"].strip(),
            "action_history": state["actions"],
            "entity_linking_results": state["entity_linking_results"] if preprocessed_sql else {},
            # In conversation history, surface only the preprocessed SQL.
            "response": (
                msg_content.replace(postprocessed_sql, preprocessed_sql)
                if postprocessed_sql
                else postprocessed_sql
            ),
        }
    )

    if enable_chainlit:
        import chainlit as cl
        cl.user_session.set("conversation_history", conversation_history)
        if suggested_next_turn_queries:
            chainlit_msg_object.actions = [
                cl.Action(name="follow-up-query", payload={"value": x}, label=x)
                for x in suggested_next_turn_queries
            ]
    elif suggested_next_turn_queries:
        msg_content += "Suggested next turn queries: " + str(suggested_next_turn_queries)

    res = {
        "agent_response": msg_content,
        "preprocessed_sql": preprocessed_sql,
        "generated_sql": postprocessed_sql,
        "conversation_history": json.dumps(conversation_history, cls=ActionEncoder),
        "result_count": result_count,
        "summary": summary,
        "sql_result": sql_result,
    }

    return _maybe_save_to_local(res, sql_query_object, save_to_local, save_result_to_csv)


def _maybe_save_to_local(
    res: Dict[str, Any],
    sql_query_object: Any,
    save_to_local: Optional[str],
    save_result_to_csv: bool,
) -> Dict[str, Any]:
    """Persist the response (and optionally a CSV of full results) to disk."""
    if not (save_to_local and os.path.isdir(save_to_local)):
        return res

    random_hash = hashlib.sha256(os.urandom(16)).hexdigest()
    file_path = os.path.join(save_to_local, f"{random_hash}.json")

    if (
        save_result_to_csv
        and sql_query_object is not None
        and getattr(sql_query_object, "execution_result_full_dict", None)
    ):
        csv_file_path = os.path.join(save_to_local, f"{random_hash}.csv")
        with open(csv_file_path, mode="w", newline="") as csv_file:
            fieldnames = list(sql_query_object.execution_result_full_dict[0].keys())
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sql_query_object.execution_result_full_dict)
        os.chmod(csv_file_path, 0o644)
        res["csv_path"] = csv_file_path
    else:
        res["csv_path"] = None

    with open(file_path, "w") as f:
        json.dump(res, f, indent=2)
    os.chmod(file_path, 0o644)
    res["file_path"] = file_path
    return res
