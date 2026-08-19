import ast
import asyncio
import concurrent.futures
import contextlib
import gc
import json
import logging
import os
import re
import uuid
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import dspy
from json_repair import repair_json
from pydantic import BaseModel

from .callback import BaseCallbackHandler
from .py_exe_utils import execute_python_script
from .storm_dataclass import DialogueTurn
from .topic_expert import TopicExpert
from .tree_search_utils import TreeNode
from ...interface import Information, Retriever
from ...langfuse_llm import call_llm_with_structured_output, get_llm
from ...rm import upload_to_azure
from ...utils import ArticleTextProcessing

script_dir = os.path.dirname(os.path.abspath(__file__))
js_dir = os.path.abspath(os.path.join(script_dir, "../../../display_javascripts"))
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

class RerankBestInsightsPreemptModel(BaseModel):
    node_id: str # random uuid
    insight: str

class RerankBestInsightsPreemptResponse(BaseModel):
    results: List[RerankBestInsightsPreemptModel]
    
class ExpandNodeResponseExploration(BaseModel):
    # Choose exactly ONE action
    is_specific_question: bool      # True if asking specific questions
    is_explore_all_columns: bool    # True if calling explore_all_columns

    # Only if is_specific_question is True
    chain_of_thought: Optional[str]
    specific_questions: Optional[List[str]]

    # Only if is_explore_all_columns is True
    explore_all_columns: Optional[str]  # SQL for explore_all_columns(sql: str)

class ExpandNodeResponseSQL(BaseModel):
    chain_of_thought: str
    sql: str
    
class ExpandNodeResponseSQLList(BaseModel):
    questions: List[ExpandNodeResponseSQL]

class ExpandNodeResponse(BaseModel):
    chain_of_thought: str
    questions: List[str]

class ExpandNodeQuestionWithDestination(BaseModel):
    question: str
    destination: str  # "internet" or "database"

class ExpandNodeResponseWithRouting(BaseModel):
    chain_of_thought: str
    questions: List[ExpandNodeQuestionWithDestination]

class ThesisCandidate(BaseModel):
    thesis: str
    research_strategy: str

class ThesisGenerationResponse(BaseModel):
    theses: List[ThesisCandidate]

class ThesisRefinementResponse(BaseModel):
    thesis: str
    research_strategy: str

class FinalInsightsConsolidationModel(BaseModel):
    consolidated_from: List[str]
    consolidated_insight: str
    query: str

class FinalInsightsConsolidationResponse(BaseModel):
    insights: List[FinalInsightsConsolidationModel]

class SummarizeModel(BaseModel):
    summary: str

class FollowUpQuestionsModel(BaseModel):
    text: str

class FinalReportResponse(BaseModel):
    report: str

class GraphGenerationResponse(BaseModel):
    python_code: str

def construct_conv(dialogue_turns: List[DialogueTurn]):
    conv = []
    for turn in dialogue_turns[:-2]:
        conv.append(
            f"Question: {turn.user_utterance}\nDB Answer: {turn.summary}"
        )
    for turn in dialogue_turns[-2:]:
        result = turn.search_results[0].snippets[0]
        conv.append(
            f"Question: {turn.user_utterance}\nDB Answer: {result}"
        )
    conv = "\n".join(conv)
    conv = conv.strip() or "N/A"
    conv = ArticleTextProcessing.limit_word_count_preserve_newline(conv, 2500)
    
    return conv

class TreeSimulator(dspy.Module):
    def __init__(
        self,
        topic_expert_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        question_asker_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retriever: Retriever,
        max_search_queries_per_turn: int,
        search_top_k: int,
        max_tree_depth = 3,
        first_level_questions: List[str] = [], # If first_level_questions is present, set the first level questions to be these
        each_level_population_control_num = 5,
        max_global_insights: int = 30,
        expansion_max_questions = 5,
        db_description = None,
        enable_followups = True,
        generate_graphs = True,
        consolidate_insights = False,
        langfuse_readonly: bool = False,
        internet_retriever: Optional[Retriever] = None,
        thesis_generation_depth: int = 3,
        thesis_refinement_interval: int = 2,
        use_global_insight_expansion: bool = True,
        skip_thesis: bool = False,
        datastorm_main_model: str = "gpt-5",
    ):
        super().__init__()
        self.datastorm_main_model = datastorm_main_model
        self.use_global_insight_expansion = use_global_insight_expansion
        self.max_tree_depth = max_tree_depth
        self.thesis_generation_depth = thesis_generation_depth
        self.thesis_refinement_interval = thesis_refinement_interval
        self.skip_thesis = skip_thesis
        self.current_thesis: Optional[str] = None
        self.current_research_strategy: Optional[str] = None
        self.last_thesis_event_completed_depth: Optional[int] = None
        self.each_level_population_control_num = each_level_population_control_num
        self.internet_retriever = internet_retriever

        # Detect whether the backing RM is DatatalkRM (the only RM whose
        # follow-up calls thread `conversation_history` through the API).
        # Use a type-name check to avoid importing the RM class here.
        _rm = getattr(retriever, "rm", None)
        self.using_datatalk_rm = type(_rm).__name__ == "DatatalkRM" if _rm is not None else False

        self.topic_expert = TopicExpert(
            engine=topic_expert_engine,
            max_search_queries=max_search_queries_per_turn,
            search_top_k=search_top_k,
            retriever=retriever,
            internet_retriever=internet_retriever,
        )
        
        self.root = TreeNode()
        self.to_expand = set([self.root])
        self.first_level_questions = first_level_questions
        
        self.global_insights = {} # {node_id: summary str}
        self.max_global_insights = max_global_insights
        self.db_description = db_description
        self.warmstart_context: Optional[str] = None
        self.expansion_max_questions = expansion_max_questions
        self.enable_followups = enable_followups
        self.generate_graphs = generate_graphs
        self.consolidate_insights = consolidate_insights
        self.langfuse_readonly = langfuse_readonly
        
        self.final_selected_res: List[DialogueTurn] = []
        self.failed_queries: List[dict] = []
        # Each entry: {type, depth, selected: {thesis, research_strategy}, candidates: [...]}
        self.thesis_events: List[dict] = []

    def save_tree_to_disk(self):
        output_dir = os.path.join(js_dir, "tree.json")
        tree_data = self.root.to_json()
        # Convenience top-level fields pointing to the latest selected thesis
        tree_data["thesis"] = self.current_thesis
        tree_data["research_strategy"] = self.current_research_strategy
        # Full audit trail of every thesis generation / refinement event
        tree_data["thesis_events"] = self.thesis_events
        tree_data["failed_queries"] = self.failed_queries
        tree_data["expansion_mode"] = "global_insight" if self.use_global_insight_expansion else "per_node"
        with open(output_dir, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, indent=2, ensure_ascii=False)
    
    async def summarize_node(self, node: TreeNode, previous_observation: str = None):
        """
        Summarize the node using the summarize_consolidate_module
        """
        llm = get_llm(model_name=self.datastorm_main_model, temperature=0)

        result = await call_llm_with_structured_output(
            "summarize",
            {
                "table_result": node.dlg_turn.search_results[0].snippets[0],
                "previous_observation": previous_observation,
                "today": datetime.now().strftime("%Y-%m-%d"),
            },
            SummarizeModel,
            llm,
            langfuse_readonly=self.langfuse_readonly,
        )
        return result.summary if result else None
    
    async def consolidate_global_insights(self, topic: str):
        """
        Consolidate global insights from the tree by consolidating
        insights from lowest level nodes + insights already in self.global_insights.
        """
        lowest_level_nodes = self.root.get_lowest_level_nodes(exclude_no_results=False)
        
        # if number of lowest level nodes + self.global_insights <= self.max_global_insights
        if len(lowest_level_nodes) + len(self.global_insights) <= self.max_global_insights:
            self.global_insights.update({
                node.node_id: node.dlg_turn.summary for node in lowest_level_nodes
            })
        else:
            llm = get_llm(model_name=self.datastorm_main_model, temperature=0)
            
            rerank_input = self.global_insights.copy()
            rerank_input.update({
                node.node_id: node.dlg_turn.summary
                for node in lowest_level_nodes
            })
            
            # Retry up to 2 times when rerank_output is None (3 attempts total)
            rerank_output = None
            for _attempt in range(3):
                candidate = await call_llm_with_structured_output(
                    "rerank_best_insights_preempt",
                    {
                        "max_num_insights": self.max_global_insights,
                        "topic": topic,
                        "input": json.dumps(rerank_input, indent=2),
                        "db_description": self.db_description,
                        "thesis": self.current_thesis,
                    },
                    RerankBestInsightsPreemptResponse,
                    llm,
                    langfuse_readonly=self.langfuse_readonly,
                )
                if candidate is not None:
                    rerank_output = candidate
                    break
                logging.warning("rerank_output was None; retrying initial rerank call...")
            if rerank_output is None:
                logging.error("rerank_output is None after retries; falling back to top max_global_insights entries")
                self.global_insights = dict(list(rerank_input.items())[:self.max_global_insights])
                return
            
            print(f"rerank_output: {rerank_output}")
            # Convert list of model objects to dict mapping node_id to summary (always use the node's own summary)
            rerank_output_dict = {result.node_id: rerank_input[result.node_id] for result in rerank_output.results if result.node_id in rerank_input}
            
            # Do a round of sanity check and attempting to fix any nodes that are not found in the tree
            not_found_nodes = []
            for key in rerank_output_dict:
                if self.root.get_node_by_id(key) is None:
                    logging.warning(f"node {key} not found in the tree")
                    not_found_nodes.append(key)
            
            if not_found_nodes:
                logging.warning(f"attempting to fix nodes that are not found in the tree: {not_found_nodes}")
                
                # Retry up to 2 times when rerank_output is None (3 attempts total)
                fixed_rerank_output = None
                for _attempt in range(3):
                    candidate = await call_llm_with_structured_output(
                        "rerank_best_insights_preempt",
                        {
                            "max_num_insights": self.max_global_insights,
                            "topic": topic,
                            "input": json.dumps(rerank_input, indent=2)
                            + "\nYou previously selected the following nodes NOT given. The list of IDs you should NOT select include: "
                            + ", ".join(not_found_nodes)
                            + ".Please fix them.",
                            "db_description": self.db_description,
                            "thesis": self.current_thesis,
                        },
                        RerankBestInsightsPreemptResponse,
                        llm,
                        langfuse_readonly=self.langfuse_readonly,
                    )
                    if candidate is not None:
                        fixed_rerank_output = candidate
                        break
                    logging.warning("rerank_output was None; retrying fix rerank call...")
                if fixed_rerank_output is not None:
                    rerank_output_dict = {result.node_id: rerank_input[result.node_id] for result in fixed_rerank_output.results if result.node_id in rerank_input}
                else:
                    logging.error("fix rerank call returned None after retries; keeping previous rerank_output_dict")
            
            self.global_insights = rerank_output_dict
            
            print(f"consolidate_global_insights: Updated global insights to a total of {len(self.global_insights)} insights")            
    
    async def generate_thesis(self, topic: str, depth: int = 0, event_type: str = "generation"):
        llm = get_llm(model_name=self.datastorm_main_model, temperature=0)

        context_parts = []
        if self.warmstart_context:
            context_parts.append(f"[Internet research from warmstart]\n{self.warmstart_context}")
        db_context = "\n\n".join(insight for insight in self.global_insights.values() if insight)
        if db_context:
            context_parts.append(f"[Database evidence from tree exploration]\n{db_context}")
        context = "\n\n".join(context_parts)

        result = await call_llm_with_structured_output(
            "generate_thesis",
            {
                "topic": topic,
                "context": context,
                "batch_idx": 0,
                "db_description": self.db_description,
            },
            ThesisGenerationResponse,
            llm,
            langfuse_readonly=self.langfuse_readonly,
        )

        if result and result.theses:
            candidates = [
                {"thesis": t.thesis, "research_strategy": t.research_strategy}
                for t in result.theses
            ]
            self.current_thesis = result.theses[0].thesis
            self.current_research_strategy = result.theses[0].research_strategy
            self.last_thesis_event_completed_depth = depth
            self.thesis_events.append({
                "type": event_type,
                "depth": depth + 1,
                "selected": candidates[0],
                "candidates": candidates,
            })
            print(f"Generated thesis: {self.current_thesis}")
            print(f"Research strategy: {self.current_research_strategy}")
        else:
            print("WARNING: Thesis generation returned no results, continuing without thesis")

    async def refine_thesis(self, topic: str, depth: int):
        llm = get_llm(model_name=self.datastorm_main_model, temperature=0)

        context_parts = []
        if self.warmstart_context:
            context_parts.append(f"[Internet research from warmstart]\n{self.warmstart_context}")
        db_context = "\n\n".join(insight for insight in self.global_insights.values() if insight)
        if db_context:
            context_parts.append(f"[Database evidence from tree exploration]\n{db_context}")
        context = "\n\n".join(context_parts)

        result = await call_llm_with_structured_output(
            "refine_thesis",
            {
                "topic": topic,
                "context": context,
                "current_thesis": self.current_thesis,
                "current_research_strategy": self.current_research_strategy,
                "db_description": self.db_description,
            },
            ThesisRefinementResponse,
            llm,
            langfuse_readonly=self.langfuse_readonly,
        )

        if result:
            refined = {"thesis": result.thesis, "research_strategy": result.research_strategy}
            self.current_thesis = result.thesis
            self.current_research_strategy = result.research_strategy
            self.last_thesis_event_completed_depth = depth
            self.thesis_events.append({
                "type": "refinement",
                "depth": depth + 1,
                "selected": refined,
                "candidates": [refined],
            })
            print(f"Refined thesis: {self.current_thesis}")
            print(f"Refined research strategy: {self.current_research_strategy}")
        else:
            print("WARNING: Thesis refinement returned no results, keeping current thesis")

    def find_reference_nodes(self, nodes: List[TreeNode]):
        """
        Find the reference nodes for the given nodes for the followup questions to clean up.
        """
        reference_nodes = []
        for node in nodes:
            reference_nodes.append(node.parent)
        return reference_nodes
    
    async def expand_once(
        self,
        topic: str,
        ground_truth_url: str,
        callback_handler: BaseCallbackHandler,
        user_utterances: List[str] = [],
        inner_expansion_max_worker = None,
        outer_expansion_max_worker = None,
        database_description = None,
    ):
        """
        Performs one BST expand operation.
        """
        next_turn_to_expand = set()
        
        llm = get_llm(model_name=self.datastorm_main_model, temperature=0)
        
        async def process_node(node):
            nonlocal user_utterances
            if not user_utterances:
                conv = construct_conv(node.reconstruct_dialog_history(lambda x: x.is_follow_up is True))
                expansion_input = {
                    "max_questions": self.expansion_max_questions,
                    "db_description": self.db_description,
                    "global_insights": json.dumps(self.global_insights, indent=2),
                    "dialogue_turns": conv,
                    "topic": topic,
                    "thesis": self.current_thesis or "",
                    "research_strategy": self.current_research_strategy or "",
                }
                
                if node.result_count != -1 and node.result_count <= 20:
                    question_prompt = "expand_node_exploration_question_gen"
                else:
                    question_prompt = "expand_node_breakdown"

                generated_list, direct_sql_gen_list = await asyncio.gather(
                    call_llm_with_structured_output(
                        question_prompt,
                        expansion_input,
                        ExpandNodeResponseWithRouting,
                        llm,
                        langfuse_readonly=self.langfuse_readonly,
                    ),
                    call_llm_with_structured_output(
                        "expand_node_exploration_direct_sql_gen",
                        expansion_input,
                        ExpandNodeResponseSQLList,
                        llm,
                        langfuse_readonly=self.langfuse_readonly,
                    ),
                )
                user_utterances = []
                for q in generated_list.questions:
                    dest = q.destination.lower().strip()
                    if dest == "internet" and self.internet_retriever is not None:
                        designation = "internet"
                    else:
                        designation = "question"  # fallback to database
                    user_utterances.append({"question": q.question, "designation": designation})
                user_utterances += [
                    {
                        "question": direct_sql_gen.sql, # this should just be a SELECT * SQL query
                        "designation": "SQL"
                    }
                    for direct_sql_gen in direct_sql_gen_list.questions
                ]
            else:
                user_utterances = [{
                    "question": user_utterance,
                    "designation": "question"
                } for user_utterance in user_utterances]

                            
            children = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=inner_expansion_max_worker) as executor:
                future_to_utterance = {
                    executor.submit(
                        self.topic_expert,
                        topic=topic,
                        question=user_utterance,
                        ground_truth_url=ground_truth_url,
                        conversation_history=None,  # Fresh query - not a follow-up to any specific conversation
                    ): user_utterance
                    for user_utterance in user_utterances
                }
            
                for future in concurrent.futures.as_completed(future_to_utterance):
                    user_utterance = future_to_utterance[future]
                    try:
                        expert_output = future.result()
                    except Exception as e:
                        logging.error(f"Failed to retrieve results for '{user_utterance['question']}': {e}")
                        print(f"WARNING: Skipping failed query: {user_utterance['question']}")
                        self.failed_queries.append({
                            "question": user_utterance["question"],
                            "designation": user_utterance.get("designation", "unknown"),
                            "depth": node.depth + 1,
                            "parent_question": node.dlg_turn.user_utterance,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        })
                        continue

                    conv_hist = (
                        expert_output.searched_results[0].conversation_history
                        if expert_output.searched_results and expert_output.searched_results[0].conversation_history
                        else node.conversation_history
                    )
                    new_node = TreeNode(
                            agent_utterance=expert_output.answer,
                            user_utterance=user_utterance["question"],
                            search_queries=expert_output.queries,
                            search_results=expert_output.searched_results,
                            conversation_history=conv_hist,
                            result_count = expert_output.searched_results[0].result_count if expert_output.searched_results else -1,
                            parent = node,
                            depth = node.depth + 1,
                        )
                    print(f"Completed processing for user utterance: {user_utterance['question']} with designation: {user_utterance['designation']}")
                    children.append(new_node)

            node.children = children
            return set(children)    

        # Use asyncio.gather to run all process_node tasks concurrently
        tasks = []
        for node in self.to_expand:
            tasks.append(process_node(node))
        
        results = await asyncio.gather(*tasks)
        children_to_cleanup = set()
        for result in results:
            children_to_cleanup = children_to_cleanup.union(result)
        
        # clean up these new level nodes by issuing followup queries
        cleaned_up_children = await self.followups_to_clean_one_level(
            list(children_to_cleanup), 
            topic, 
            ground_truth_url,
            enable_followups=self.enable_followups,
        )
        
        next_turn_to_expand = set(cleaned_up_children)
    
        self.save_tree_to_disk()
        
        print(f"Generated a total of {len(next_turn_to_expand)} new frontier nodes from {len(self.to_expand)} frontier nodes")
        self.to_expand = next_turn_to_expand
        
        # Run all async operations concurrently
        await self.consolidate_global_insights(topic)
        
        # Filter frontier nodes
        prior_to_filter_len = len(self.to_expand)
        self.to_expand = await self.listwise_llm_rerank(
            topic,
            self.to_expand,
            cut_off_num=self.each_level_population_control_num
        )
        post_to_filter_len = len(self.to_expand)
        if list(self.to_expand):
            print(f"Filtering depth {list(self.to_expand)[0].depth} nodes. Filtered from {prior_to_filter_len} to {post_to_filter_len} number of nodes")
        else:
            print(f"Filtering depth N/A nodes (no available nodes). Filtered from {prior_to_filter_len} to {post_to_filter_len} number of nodes")
    
    async def expand_from_global_insights(
        self,
        topic: str,
        ground_truth_url: str,
        callback_handler: BaseCallbackHandler,
        current_depth: int,
        inner_expansion_max_worker=None,
    ):
        """
        Generates the next batch of questions using only the global insight bank,
        thesis, and topic — bypassing per-node conversation history entirely.
        All new nodes are children of self.root.
        """
        llm = get_llm(model_name=self.datastorm_main_model, temperature=0)

        expansion_input = {
            "max_questions": self.expansion_max_questions,
            "db_description": self.db_description,
            "global_insights": json.dumps(self.global_insights, indent=2),
            "dialogue_turns": "",
            "topic": topic,
            "thesis": self.current_thesis or "",
            "research_strategy": self.current_research_strategy or "",
        }

        generated_list, direct_sql_gen_list = await asyncio.gather(
            call_llm_with_structured_output(
                "expand_node_breakdown",
                expansion_input,
                ExpandNodeResponseWithRouting,
                llm,
                langfuse_readonly=self.langfuse_readonly,
            ),
            call_llm_with_structured_output(
                "expand_node_exploration_direct_sql_gen",
                expansion_input,
                ExpandNodeResponseSQLList,
                llm,
                langfuse_readonly=self.langfuse_readonly,
            ),
        )

        # call_llm_with_structured_output returns None when the call fails —
        # most often Azure's content filter rejecting war-related text, which is
        # effectively unavoidable over a long run. Dereferencing None here used
        # to abort the whole run mid-tree (AttributeError: 'NoneType' object has
        # no attribute 'questions'), discarding hours of completed research.
        # Degrade to "this node produced no expansion" instead; the two calls are
        # independent, so one failing must not discard the other's output.
        if generated_list is None:
            logging.warning(
                "expand_from_global_insights: 'expand_node_breakdown' returned no "
                "output (LLM call failed); skipping question expansion for this node."
            )
        if direct_sql_gen_list is None:
            logging.warning(
                "expand_from_global_insights: 'expand_node_exploration_direct_sql_gen' "
                "returned no output (LLM call failed); skipping direct-SQL expansion "
                "for this node."
            )
        if generated_list is None and direct_sql_gen_list is None:
            return

        user_utterances = []
        for q in (generated_list.questions if generated_list is not None else []):
            dest = q.destination.lower().strip()
            if dest == "internet" and self.internet_retriever is not None:
                designation = "internet"
            else:
                designation = "question"
            user_utterances.append({"question": q.question, "designation": designation})
        user_utterances += [
            {"question": direct_sql_gen.sql, "designation": "SQL"}
            for direct_sql_gen in (
                direct_sql_gen_list.questions if direct_sql_gen_list is not None else []
            )
        ]

        children = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=inner_expansion_max_worker) as executor:
            future_to_utterance = {
                executor.submit(
                    self.topic_expert,
                    topic=topic,
                    question=user_utterance,
                    ground_truth_url=ground_truth_url,
                    conversation_history=None,
                ): user_utterance
                for user_utterance in user_utterances
            }

            for future in concurrent.futures.as_completed(future_to_utterance):
                user_utterance = future_to_utterance[future]
                try:
                    expert_output = future.result()
                except Exception as e:
                    logging.error(f"Failed to retrieve results for '{user_utterance['question']}': {e}")
                    print(f"WARNING: Skipping failed query: {user_utterance['question']}")
                    self.failed_queries.append({
                        "question": user_utterance["question"],
                        "designation": user_utterance.get("designation", "unknown"),
                        "depth": current_depth + 2,
                        "parent_question": self.root.dlg_turn.user_utterance,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    })
                    continue

                conv_hist = (
                    expert_output.searched_results[0].conversation_history
                    if expert_output.searched_results and expert_output.searched_results[0].conversation_history
                    else None
                )
                new_node = TreeNode(
                    agent_utterance=expert_output.answer,
                    user_utterance=user_utterance["question"],
                    search_queries=expert_output.queries,
                    search_results=expert_output.searched_results,
                    conversation_history=conv_hist,
                    result_count=expert_output.searched_results[0].result_count if expert_output.searched_results else -1,
                    parent=self.root,
                    depth=current_depth + 2,
                )
                print(f"Completed processing for user utterance: {user_utterance['question']} with designation: {user_utterance['designation']}")
                children.append(new_node)

        self.root.children.extend(children)

        cleaned_up_children = await self.followups_to_clean_one_level(
            children,
            topic,
            ground_truth_url,
            enable_followups=self.enable_followups,
        )

        print(f"Generated {len(cleaned_up_children)} new nodes from global insight-driven expansion at depth {current_depth + 2}")

        self.save_tree_to_disk()
        await self.consolidate_global_insights(topic)

    async def listwise_llm_rerank(self, topic: str, nodes: List[TreeNode], cut_off_num: int):
        """
        Rerank the nodes by feeding in a list to an LLM.
        """
        llm = get_llm(model_name=self.datastorm_main_model, temperature=0)
        
        listwise_llm_rerank_input = {
            node.node_id: node.dlg_turn.summary
            for node in nodes if node.dlg_turn.summary
        }

        listwise_llm_rerank_output = await call_llm_with_structured_output(
            "rerank_best_insights_preempt",
            {
                "max_num_insights": cut_off_num,
                "topic": topic,
                "input": json.dumps(listwise_llm_rerank_input, indent=2),
                "db_description": self.db_description,
                "thesis": self.current_thesis,
            },
            RerankBestInsightsPreemptResponse,
            llm,
            langfuse_readonly=self.langfuse_readonly,
        )
            
        # The rerank LLM call can return None (e.g. a content-policy violation on
        # sensitive conflict topics, or repeated parse failure). Don't crash the
        # whole run on None.results — fall back to keeping the first cut_off_num
        # frontier nodes so expansion continues.
        if listwise_llm_rerank_output is None or not getattr(listwise_llm_rerank_output, "results", None):
            logging.warning(
                "listwise_llm_rerank returned no usable output (LLM None / content policy?); "
                "falling back to the first %d nodes.", cut_off_num
            )
            return nodes[:cut_off_num]

        res = [result.node_id for result in listwise_llm_rerank_output.results]
        res_nodes = [node for node in nodes if node.node_id in res]

        return res_nodes
    
        
    async def followups_to_clean_one_level(
        self,
        nodes: List[TreeNode],
        topic: str,
        ground_truth_url: str,
        enable_followups = False,
    ):
        """
        Sometimes, one node was unable to return results but its peer nodes were able to return results.
        This could be due to Datatalk not following all instructions in the current node (but its peer node was able to).
        For instance, in the current node it might just do (actor1 LIKE '%ISIS%' OR actor2 LIKE '%ISIS%') to filter ISIS activities, but the correct one would be to use the entity_linking module.
        Probabilistically, most of the nodes can, but this raises the question of how to make them consistent on the same level w.r.t. the filters being used.
        
        This function takes a look at all the reported SQLs and attempt to consolidate them.
        
        Returns:
            a new list of TreeNodes that are follow-up questions to their parents.
        """
        
        # Helper function to remove consecutive duplicates
        def remove_consecutive_duplicates(items):
            result = []
            for i, item in enumerate(items):
                if i == 0 or item != items[i-1]:
                    result.append(item)
            return result
        
        def get_reference_nodes_json():
            result = {}

            for i, (node_id, _) in enumerate(self.global_insights.items()):
                node = self.root.get_node_by_id(node_id)
                if node is None or not node.dlg_turn.search_results:
                    continue
                
                # if the node is a follow-up question, include the parent question in the query
                query = node.parent.dlg_turn.user_utterance + "\n" + node.dlg_turn.user_utterance if node.is_follow_up else node.dlg_turn.user_utterance
                result[f"example_node_{i}"] = {
                    "query": query,
                    "SQL": node.dlg_turn.search_results[0].meta.get("preprocessed_sql", "N/A"),
                    "example_node": True,
                    "note": "no need to generate follow_up_question, shown here only as reference"
                }
            return result
            
            
        llm = get_llm(model_name=self.datastorm_main_model, temperature=0)
        nodes_look_up_dict = {
            f"query{i}": node
            for i, node in enumerate(nodes)
        }
        all_new_nodes = {}
        
        if enable_followups:
            input_json = get_reference_nodes_json()
            
            input_json.update({
                f"query{i}": {
                    "previous_queries": remove_consecutive_duplicates([node.user_utterance for node in node.reconstruct_dialog_history()]),
                    "query": node.dlg_turn.user_utterance,
                        "SQL": node.dlg_turn.search_results[0].meta.get("preprocessed_sql", "N/A") if node.dlg_turn.search_results else None
                    }
                    for i, node in enumerate(nodes)
                }
            )
            
            result = await call_llm_with_structured_output(
                "followups_to_clean_one_level",
                {
                    "input": json.dumps(input_json, indent=2),
                },
                FollowUpQuestionsModel,
                llm,
                langfuse_readonly=self.langfuse_readonly,
            )
            res_text = result.text if result else "{}"
            try:
                res = json.loads(res_text)
            except Exception as e:
                logging.warning(f"JSON parse error in followups_to_clean_one_level, attempting repair: {e}")
                try:
                    res = json.loads(repair_json(res_text))
                except Exception as e2:
                    logging.error(f"JSON repair also failed, neglecting follow-up questions: {e2}")
                    res = {}

            # The prompt should return a JSON object keyed by node id, but the LLM
            # occasionally returns a JSON array (or other non-object). That would
            # crash the whole run at res.items(); degrade gracefully instead by
            # skipping follow-ups for this level, mirroring the parse-failure path.
            if not isinstance(res, dict):
                logging.warning(
                    "followups_to_clean_one_level returned a %s, not an object; "
                    "skipping follow-ups for this level.", type(res).__name__
                )
                res = {}

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_key = {
                    executor.submit(
                        self.topic_expert,
                        topic=topic,
                        question=value["follow_up_question"],
                        ground_truth_url=ground_truth_url,
                        conversation_history=nodes_look_up_dict[key].conversation_history
                    ): key
                    for key, value in res.items()
                    if value and value.get("follow_up_question") and key in nodes_look_up_dict
                }

                # Process the futures for the if branch
                for future in concurrent.futures.as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        expert_output = future.result()
                        
                        searched_results : List[Information] = expert_output.searched_results
                        if searched_results:
                            searched_results[0].meta["query"] = nodes_look_up_dict[key].dlg_turn.user_utterance

                        user_utterance = res[key]["follow_up_question"]
                        new_node = TreeNode(
                            agent_utterance=expert_output.answer,
                            user_utterance=user_utterance,
                            search_queries=expert_output.queries,
                            search_results=searched_results,
                            conversation_history=searched_results[0].conversation_history if searched_results else nodes_look_up_dict[key].conversation_history,
                            result_count=searched_results[0].result_count if searched_results else -1,
                            parent=nodes_look_up_dict[key],
                            depth=nodes_look_up_dict[key].depth + 1,
                        )
                        print(f"Completed processing for user utterance: {user_utterance}")
                        all_new_nodes[key] = new_node
                    except Exception as e:
                        logging.error(f"Error processing node {key}: {e}")
        else:
            # constructs a dictionary where all follow-up questions are None
            res = {
                key: {
                    "follow_up_question": None,
                }
                for key, value in nodes_look_up_dict.items()
            }

        # process the nodes that can be skipped (LLM outputs None)
        for key, value in res.items():
            if (value is None or not value.get("follow_up_question")) and key in nodes_look_up_dict:
                new_node = deepcopy(nodes_look_up_dict[key])
                new_node.parent = nodes_look_up_dict[key]
                new_node.depth += 1
                new_node.regenerate_node_id()
                all_new_nodes[key] = new_node
                
        for key, new_node in all_new_nodes.items():
            # Summarize the node from its SQL result. This is REQUIRED regardless of
            # enable_followups: the global-insight selection (consolidate_global_insights
            # rerank + final_consolidate_insights) ranks/consolidates node summaries, so
            # without a summary the selection runs blind and discards the node's DB
            # evidence. The QC ablation (--disable_followups) must not also disable
            # summarization. The `not summary` guard keeps cost bounded: with QC off a
            # node is a deepcopy of its parent and inherits the parent's summary, so each
            # distinct node is summarized exactly once.
            if new_node.dlg_turn.search_results and not new_node.dlg_turn.summary:
                if new_node.parent and new_node.parent.parent and new_node.parent.parent.dlg_turn.summary:
                    context = new_node.parent.parent.dlg_turn.summary
                else:
                    context = "No previous observation"

                summary = await self.summarize_node(new_node, previous_observation=context)
                new_node.dlg_turn.summary = summary
            new_node.is_follow_up = True

            nodes_look_up_dict[key].children = [new_node]
        
        return list(all_new_nodes.values())
        
    async def generate_graph_from_node_matplotlib(self, node: TreeNode):
        """
        Generate a matplotlib PNG plot for a given node using the sandbox-execution visualization approach.
        Uses a persistent docker container for faster execution.

        Args:
            node (TreeNode): The node to generate the plot for.

        Returns:
            str: The path to the generated PNG file.
        """
        import base64
        import docker

        llm = get_llm(model_name=self.datastorm_main_model, temperature=0)

        graph_result = await call_llm_with_structured_output(
            "graph_generation_matplotlib",
            {
                "csv_file_path": node.dlg_turn.search_results[0].meta["csv_path"],
                "csv_snapshot": node.dlg_turn.search_results[0].snippets[0],
            },
            GraphGenerationResponse,
            llm,
            langfuse_readonly=self.langfuse_readonly,
        )
        python_code = (graph_result.python_code if graph_result else "").strip()
        # Remove markdown code block markers if present
        if python_code.startswith("```python"):
            python_code = python_code[len("```python"):].strip()
        elif python_code.startswith("```"):
            python_code = python_code[len("```"):].strip()

        if python_code.endswith("```"):
            python_code = python_code[:-len("```")].strip()

        # Inject matplotlib hook similar to the datatalk_agent sandbox approach
        injected_code = (
            "import base64, io, sys\n"
            "import pandas as pd\n"
            "import warnings\n"
            "warnings.filterwarnings('ignore', '.*numpy.core.numeric.*', DeprecationWarning)\n"
            "try:\n"
            "    import matplotlib\n"
            "    matplotlib.use('Agg')\n"
            "    import matplotlib.pyplot as plt\n"
            "    def __storm_emit_plot(fig):\n"
            "        buf = io.BytesIO()\n"
            "        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')\n"
            "        buf.seek(0)\n"
            "        data = base64.b64encode(buf.getvalue()).decode('ascii')\n"
            "        sys.stdout.write('<<STORM_PLOT:PNG>>' + data + '<<END>>\\n')\n"
            "        sys.stdout.flush()\n"
            "    def __storm_show(*args, **kwargs):\n"
            "        try:\n"
            "            import matplotlib._pylab_helpers as pylab_helpers\n"
            "            managers = pylab_helpers.Gcf.get_all_fig_managers()\n"
            "            if not managers:\n"
            "                return\n"
            "            for m in managers:\n"
            "                __storm_emit_plot(m.canvas.figure)\n"
            "        except Exception as e:\n"
            "            sys.stdout.write(f'<<STORM_PLOT_ERROR>>{e}<<END>>\\n')\n"
            "            sys.stdout.flush()\n"
            "    # Monkey-patch show so any plt.show() in user code triggers our emitter\n"
            "    plt.show = __storm_show\n"
            "except Exception:\n"
            "    # If matplotlib is unavailable, proceed without hooking\n"
            "    pass\n"
            f"{python_code}\n"
        )

        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../python_executions')
        # Create directory for saving code and results if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        csv_file_path = node.dlg_turn.search_results[0].meta["csv_path"]
        if not csv_file_path:
            print(f"CSV file for {node.dlg_turn.user_utterance} does not exist")
            return None
        execution_id = os.path.splitext(csv_file_path)[0].split("/")[-1]

        # Save the Python code to a file
        code_file_path = os.path.join(save_dir, f'code_matplotlib_{execution_id}.py')
        with open(code_file_path, 'w') as f:
            f.write(f"# Associated CSV file: {node.dlg_turn.search_results[0].meta['csv_path']}\n\n")
            f.write(python_code)

        # Execute in persistent docker container (datatalk_agent-style)
        _docker_ok = False
        try:
            client = docker.from_env()
            container_name = "llm-sandbox-datastorm"

            exec_result = client.containers.get(container_name).exec_run(
                cmd=["python", "-c", injected_code],
                user="sandbox",  # enforce non-root execution
                stdout=True,
                stderr=True
            )
            output = exec_result.output.decode()
            # exec_run does not raise on non-zero exit or cgroup/OCI errors —
            # detect failure by exit code or known error patterns in the output.
            if exec_result.exit_code != 0 or "OCI runtime exec failed" in output or "no such file or directory" in output:
                raise RuntimeError(f"Docker exec failed (exit={exec_result.exit_code}): {output[:300]}")
            _docker_ok = True
        except Exception as e:
            print(f"Error executing in docker container: {e}")
            print("Falling back to execute_python_script")
            # Fallback to the old method if docker fails
            res = execute_python_script(injected_code)
            output = res["stdout"] + res["stderr"]

        # Save the output to a file
        output_file_path = os.path.join(save_dir, f'output_matplotlib_{execution_id}.txt')
        with open(output_file_path, 'w') as f:
            f.write(f'# Associated CSV file: {node.dlg_turn.search_results[0].meta["csv_path"]}\n\n')
            f.write(f"# Code executed:\n{python_code}\n\n")
            f.write(f"# Output:\n{output}")

        print(f"Python code and output saved to {code_file_path}")

        # Extract the base64-encoded PNG from the output
        png_file_path = None
        if '<<STORM_PLOT:PNG>>' in output:
            try:
                # Extract base64 data between sentinels
                start_marker = '<<STORM_PLOT:PNG>>'
                end_marker = '<<END>>'
                start_idx = output.index(start_marker) + len(start_marker)
                end_idx = output.index(end_marker, start_idx)
                base64_data = output[start_idx:end_idx].strip()

                # Decode and save as PNG
                png_data = base64.b64decode(base64_data)

                # Ensure the csv_file_path ends with .csv before replacing
                if csv_file_path.endswith(".csv"):
                    png_file_path = csv_file_path.replace(".csv", ".png")
                else:
                    print(f"Warning: CSV file {csv_file_path} does not end with .csv")
                    png_file_path = csv_file_path + ".png"

                with open(png_file_path, 'wb') as f:
                    f.write(png_data)

                print(f"PNG file saved to {png_file_path}")
                return png_file_path
            except Exception as e:
                print(f"Error extracting PNG from output: {e}")
                return None
        else:
            print(f"No PNG plot found in output for {node.dlg_turn.user_utterance}")
            return None

    async def generate_graph_from_node(self, node: TreeNode, use_matplotlib: bool = True):
        """
        Generate a plot for a given node. Can use either plotly (HTML) or matplotlib (PNG).

        Args:
            node (TreeNode): The node to generate the plot for.
            use_matplotlib (bool): If True, use matplotlib to generate PNG. If False, use plotly to generate HTML.

        Returns:
            str: The path to the generated file (HTML for plotly, PNG for matplotlib).
        """
        # Skip nodes without csv_path (e.g. internet-sourced nodes)
        if (not node.dlg_turn.search_results
            or "csv_path" not in node.dlg_turn.search_results[0].meta):
            print(f"Skipping graph generation for '{node.dlg_turn.user_utterance}': no csv_path in meta")
            return None

        if use_matplotlib:
            return await self.generate_graph_from_node_matplotlib(node)

        # Original plotly implementation
        llm = get_llm(model_name=self.datastorm_main_model, temperature=0)

        graph_result = await call_llm_with_structured_output(
            "graph_generation",
            {
                "csv_file_path": node.dlg_turn.search_results[0].meta["csv_path"],
                "csv_snapshot": node.dlg_turn.search_results[0].snippets[0],
            },
            GraphGenerationResponse,
            llm,
            langfuse_readonly=self.langfuse_readonly,
        )
        python_code = (graph_result.python_code if graph_result else "").strip()
        # Remove markdown code block markers if present
        if python_code.startswith("```python"):
            python_code = python_code[len("```python"):].strip()
        elif python_code.startswith("```"):
            python_code = python_code[len("```"):].strip()
            
        if python_code.endswith("```"):
            python_code = python_code[:-len("```")].strip()
        
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../python_executions')
        # Create directory for saving code and results if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        csv_file_path = node.dlg_turn.search_results[0].meta["csv_path"]
        if not csv_file_path:
            print(f"CSV file for {node.dlg_turn.user_utterance} does not exist")
            return None
        execution_id = os.path.splitext(csv_file_path)[0].split("/")[-1]
        # Ensure the csv_file_path ends with .csv before replacing
        if csv_file_path.endswith(".csv"):
            html_file_path = csv_file_path.replace(".csv", ".html")
        else:
            print(f"Warning: CSV file {csv_file_path} does not end with .csv")
            html_file_path = csv_file_path + ".html"
        
        # Save the Python code to a file
        code_file_path = os.path.join(save_dir, f'code_{execution_id}.py')
        with open(code_file_path, 'w') as f:
            f.write(f"# Associated CSV file: {node.dlg_turn.search_results[0].meta['csv_path']}\n\n")
            f.write(python_code)
        
        res = execute_python_script(python_code)
        output = res["stdout"] + res["stderr"]
        
        # Save the output to a file
        output_file_path = os.path.join(save_dir, f'output_{execution_id}.txt')
        with open(output_file_path, 'w') as f:
            f.write(f'# Associated CSV file: {node.dlg_turn.search_results[0].meta["csv_path"]}\n\n')
            f.write(f"# Code executed:\n{python_code}\n\n")
            f.write(f"# Output:\n{output}")
        
        print(f"Python code and output saved to {code_file_path}")
        
        # check if the HTML file has been generated
        
        if not os.path.exists(html_file_path):
            print(f"HTML file {html_file_path} not found")
            return None
        else:
            return html_file_path
    
    async def final_consolidate_insights(self):
        """
            Consolidate the insights from the global insights at the end
            
            The goal is to consolidate any insights that can be combined into one SQL result.
        """
        llm = get_llm(model_name=self.datastorm_main_model, temperature=0)
        
        insights_json = {}
        i = 0
        insight_node_mapping = {}
        for node_id, insight in self.global_insights.items():
            insights_json_key = f"insight_{i}"
            node = self.root.get_node_by_id(node_id)
            insight_node_mapping[insights_json_key] = node
            
            insights_json_value = {
                "summary": node.dlg_turn.summary,
                "sql": node.dlg_turn.search_results[0].meta.get("preprocessed_sql", "N/A"),
                "table_result": node.dlg_turn.search_results[0].snippets[0]
            }
            insights_json[insights_json_key] = insights_json_value
            i += 1
        
        res = await call_llm_with_structured_output(
            "final_insights_consolidation",
            {
                "insights": json.dumps(insights_json, indent=2),
            },
            FinalInsightsConsolidationResponse,
            llm,
            langfuse_readonly=self.langfuse_readonly,
        )
        
        print(res.insights)
        print(f"final_consolidate_insights: consolidated from {len(self.global_insights)} insights to {len(res.insights)} insights")
        self.global_insights = {}
        
        # Keep track of which insights from the original mapping are used
        used_insight_keys = set()
        for insight in res.insights:
            # Check if any consolidated_from IDs are not in the insight_node_mapping
            invalid_insight_ids = [insight_id for insight_id in insight.consolidated_from if insight_id not in insight_node_mapping]
            if invalid_insight_ids:
                print(f"WARNING: The following insight IDs were not found in the mapping: {invalid_insight_ids}")
                # Keep only the valid IDs
                valid_insight_ids = [insight_id for insight_id in insight.consolidated_from if insight_id in insight_node_mapping]
                if not valid_insight_ids:
                    print(f"ERROR: No valid insights found in {insight.consolidated_from}, skipping this consolidation")
                    continue
                insight.consolidated_from = valid_insight_ids
                
            # Track which insight keys are used
            used_insight_keys.update(insight.consolidated_from)
            
            nodes : List[TreeNode] = [insight_node_mapping[insight_id] for insight_id in insight.consolidated_from]
            
            if len(nodes) == 1:
                self.global_insights[nodes[0].node_id] = nodes[0].dlg_turn.summary
                nodes[0].is_final_selected = True

            elif len(nodes) > 1:
                sql_queries = []
                for node in nodes:
                    sql_queries.append(node.dlg_turn.search_results[0].meta.get("preprocessed_sql", "N/A"))
                
                query = f"Combine the following SQL results into one: {sql_queries}. Give human readable column names (as opposed to just say the column was from the first query)"
                # print(f"retrieving {query}")
                conversation_history = []
                if self.using_datatalk_rm:
                    for node in nodes:
                        if node.conversation_history:
                            if type(node.conversation_history) == str:
                                node_conversation_history = json.loads(node.conversation_history)
                            else:
                                node_conversation_history = node.conversation_history
                            assert type(node_conversation_history) == list
                            conversation_history.extend(node_conversation_history)
                
                    searched_results: List[Information] = self.topic_expert.retriever.retrieve(
                        [query],
                        conversation_history=json.dumps(conversation_history)
                    )
                else:
                    searched_results: List[Information] = self.topic_expert.retriever.retrieve(
                        [query],
                    )
                
                if not searched_results:
                    print(f"WARNING: Failed to retrieve combined results for nodes: {[node.node_id for node in nodes]}. Adding individual insights instead.")
                    for node in nodes:
                        self.global_insights[node.node_id] = node.dlg_turn.summary
                        node.is_final_selected = True
                else:
                    # construct a new node to store the consolidated insight
                    node_ids = [node.node_id for node in nodes]
                    new_node = TreeNode(
                        user_utterance=query,
                        search_queries=[query],
                        search_results=searched_results,
                        result_count=searched_results[0].result_count if searched_results else -1,
                        is_final_selected=True,
                        parent=nodes[0],
                        depth=nodes[0].depth + 1
                    )
                    nodes[0].children += [new_node]
                    context = "This new result is based on a list of previous observations: " + ";".join([node.dlg_turn.summary for node in nodes])
                    new_node.dlg_turn.summary = await self.summarize_node(new_node, previous_observation=context)
                    new_node.dlg_turn.agent_utterance = new_node.dlg_turn.summary
                    new_node.dlg_turn.search_results[0].meta["consolidate_info"] = f"based on consolidated insights from {', '.join(node_ids)}. LLM-predicted consolidated insight (this is not used anywhere, just for information purpose): {insight.consolidated_insight}"
                    
                    # This query field is used by Co-STORM in its organization phase
                    new_node.dlg_turn.search_results[0].meta["query"] = insight.query
                    self.global_insights[new_node.node_id] = new_node.dlg_turn.summary
            
            else:
                print(f"final_consolidate_insights: node {insight.consolidated_from} not found in the tree")
        
        # Add any insights that weren't included in the consolidated results
        unused_insight_keys = set(insight_node_mapping.keys()) - used_insight_keys
        if unused_insight_keys:
            print(f"Found {len(unused_insight_keys)} insights that weren't included in the consolidated results. Adding them as individual insights.")
            for key in unused_insight_keys:
                node = insight_node_mapping[key]
                if node and node.dlg_turn and node.dlg_turn.summary:
                    self.global_insights[node.node_id] = node.dlg_turn.summary
                    node.is_final_selected = True
                    print(f"Added unused insight: {key} (node_id: {node.node_id})")
                else:
                    print(f"WARNING: Could not add unused insight {key} as it has invalid data")
        print(f"final_consolidate_insights: {len(self.global_insights)} insights left")
    
    async def forward(
        self,
        topic: str,
        ground_truth_url: str,
        callback_handler: BaseCallbackHandler,
    ):
        """
            Performs the BST expansion up to self.max_tree_depth.
            Depth 0 → 1 uses first_level_questions via expand_once.
            Depth 1 → max_tree_depth uses global insight-driven expansion.
        """
        # Depth 0 → 1: seed expansion with first-level questions
        print("Expanding depth 0 nodes")
        await self.expand_once(
            topic,
            ground_truth_url,
            callback_handler,
            user_utterances=self.first_level_questions,
        )

        completed_depth = 1
        if not self.skip_thesis:
            if completed_depth == self.thesis_generation_depth and self.current_thesis is None:
                await self.generate_thesis(topic, depth=completed_depth)
            elif (
                self.current_thesis is not None
                and self.last_thesis_event_completed_depth is not None
                and completed_depth - self.last_thesis_event_completed_depth >= self.thesis_refinement_interval
            ):
                await self.refine_thesis(topic, depth=completed_depth)

        # Depth 1 → max_tree_depth: global insight-driven or per-node expansion
        if self.use_global_insight_expansion:
            for depth in range(1, self.max_tree_depth):
                if not self.global_insights:
                    print("No global insights available; stopping expansion.")
                    break
                print(f"Expanding depth {depth} nodes via global insights")
                await self.expand_from_global_insights(
                    topic,
                    ground_truth_url,
                    callback_handler,
                    current_depth=depth,
                )

                completed_depth = depth + 1
                if not self.skip_thesis:
                    if completed_depth == self.thesis_generation_depth and self.current_thesis is None:
                        await self.generate_thesis(topic, depth=completed_depth)
                    elif (
                        self.current_thesis is not None
                        and self.last_thesis_event_completed_depth is not None
                        and completed_depth - self.last_thesis_event_completed_depth >= self.thesis_refinement_interval
                    ):
                        await self.refine_thesis(topic, depth=completed_depth)
        else:
            for depth in range(1, self.max_tree_depth):
                if not self.to_expand:
                    print("No frontier nodes to expand; stopping expansion.")
                    break
                print(f"Expanding depth {depth} nodes via per-node expansion")
                await self.expand_once(topic, ground_truth_url, callback_handler)

                completed_depth = depth + 1
                if not self.skip_thesis:
                    if completed_depth == self.thesis_generation_depth and self.current_thesis is None:
                        await self.generate_thesis(topic, depth=completed_depth)
                    elif (
                        self.current_thesis is not None
                        and self.last_thesis_event_completed_depth is not None
                        and completed_depth - self.last_thesis_event_completed_depth >= self.thesis_refinement_interval
                    ):
                        await self.refine_thesis(topic, depth=completed_depth)

        if self.consolidate_insights:
            await self.final_consolidate_insights()
        
        selected_res = []
        # Create tasks for parallel processing
        node_tasks = []
        for node_id, insight in self.global_insights.items():
            node = self.root.get_node_by_id(node_id)
            if node is None:
                logging.warning(f"node {node_id} not found in the tree")
            else:
                node.is_final_selected = True
                node_tasks.append((node, node_id))
        
        # Process all generate_graph_from_node calls in parallel
        if node_tasks and self.generate_graphs:
            async def process_node(node):
                html_file_path = await self.generate_graph_from_node(node)
                if html_file_path:
                    # The upload only turns an already-generated local artifact
                    # into a shareable URL. Letting it raise aborts the whole
                    # run from inside asyncio.gather, discarding every node —
                    # which is what an expired SAS token used to do here. Keep
                    # the local path on failure instead. Note TreeSimulator is
                    # never given `disable_upload_to_azure`, so this path runs
                    # even when the CLI flag is set; that is why it must not be
                    # fatal.
                    try:
                        azure_html_file_path = upload_to_azure(html_file_path)
                    except Exception as e:
                        logging.warning(
                            "upload_to_azure failed for graph %s; keeping local path. Error: %s",
                            html_file_path,
                            repr(e)[:200],
                        )
                        azure_html_file_path = html_file_path
                    node.dlg_turn.search_results[0].meta["html_file_path"] = azure_html_file_path
                    # node.dlg_turn.search_results[0].meta["sql_result"] += f"\n\n[Click here for visualization]({azure_html_file_path})"
                return node.dlg_turn
            
            # Run all visualization tasks concurrently
            tasks = [process_node(node) for node, _ in node_tasks]
            dlg_turns = await asyncio.gather(*tasks)
            selected_res.extend(dlg_turns)
        else:
            if not self.generate_graphs:
                print("Skipping graph generation as requested")
            selected_res = [node.dlg_turn for node, _ in node_tasks]
        
        self.save_tree_to_disk()
        
        self.final_selected_res = selected_res
        return dspy.Prediction(dlg_history=selected_res)
    
    # For compatibility with existing code that expects a sync function
    def __call__(self, *args, **kwargs):
        loop = None
        try:
            # Configure asyncio to use a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the coroutine and get the result
            result = loop.run_until_complete(self.forward(*args, **kwargs))
            
            return result
        except KeyboardInterrupt:
            print("\nInterrupted by user. Cleaning up...")
            # Re-raise to allow normal keyboard interrupt handling
            raise
        finally:
            # Close the loop properly if it exists
            if loop:
                try:
                    # Force garbage collection to help identify lingering objects
                    gc.collect()
                    
                    # Find and properly close any HTTP client sessions
                    self._cleanup_http_clients(loop)
                    
                    # Clean up all tasks
                    self._cleanup_tasks(loop)
                    
                    # Shutdown async generators
                    with contextlib.suppress(Exception):
                        loop.run_until_complete(loop.shutdown_asyncgens())
                    
                    # Shutdown default executor if possible
                    with contextlib.suppress(Exception):
                        if hasattr(loop, 'shutdown_default_executor'):
                            loop.run_until_complete(loop.shutdown_default_executor())
                    
                    # Close the loop
                    loop.close()
                except Exception as e:
                    print(f"Error during cleanup: {e}")
                finally:
                    # Reset the event loop
                    asyncio.set_event_loop(None)
    
    def _cleanup_http_clients(self, loop):
        """Find and properly close any HTTP clients."""
        # This is a best-effort attempt to clean up HTTP clients
        with contextlib.suppress(Exception):
            # Look for httpx.AsyncClient instances and close them
            for obj in gc.get_objects():
                # Check if it looks like an httpx client
                if hasattr(obj, 'aclose') and callable(obj.aclose) and 'httpx' in str(type(obj)):
                    try:
                        # Silence resource warnings during cleanup
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ResourceWarning)
                            # Create a new task just for closing this client
                            close_task = loop.create_task(obj.aclose())
                            loop.run_until_complete(close_task)
                    except Exception:
                        pass  # Best effort only
    
    def _cleanup_tasks(self, loop):
        """Cancel all running tasks in the loop."""
        with contextlib.suppress(Exception):
            tasks = asyncio.all_tasks(loop)
            if tasks:
                # Cancel all tasks
                for task in tasks:
                    task.cancel()
                
                # Wait for tasks to be cancelled
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

