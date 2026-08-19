"""Knowledge curation orchestration.

Drives the BST tree search (TreeSimulator) and exposes the result as a
StormInformationTable. The heavy classes live in sibling modules:

  tree_simulator.py  — TreeSimulator and the pydantic response models
                       used by its expand/thesis/consolidate prompts.
  topic_expert.py    — TopicExpert, a thin retriever-routing wrapper
                       used by TreeSimulator inside expand_once().

External callers (notably the entry point) historically imported
TreeSimulator from this module, so it is re-exported below.
"""

import concurrent.futures
from typing import Dict, Optional, Tuple, Union

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable
from .topic_expert import TopicExpert
from .tree_simulator import TreeSimulator
from ...interface import KnowledgeCurationModule, Retriever


__all__ = [
    "StormKnowledgeCurationModule",
    "TreeSimulator",
    "TopicExpert",
]


class StormKnowledgeCurationModule(KnowledgeCurationModule):
    """Knowledge curation stage: drive a TreeSimulator BST tree search and
    surface the selected dialog turns plus the serialized tree."""

    def __init__(
        self,
        retriever: Retriever,
        conv_simulator_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        question_asker_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        max_search_queries_per_turn: int,
        search_top_k: int,
        max_thread_num: int,
        max_tree_depth: int = 3,
        first_level_questions=None,
        each_level_population_control_num: int = 5,
        max_global_insights: int = 30,
        db_description: Optional[str] = None,
        expansion_max_questions: int = 5,
        enable_followups: bool = True,
        generate_graphs: bool = True,
        consolidate_insights: bool = False,
        langfuse_readonly: bool = False,
        internet_retriever: Optional[Retriever] = None,
        thesis_generation_depth: int = 3,
        thesis_refinement_interval: int = 2,
        skip_thesis: bool = False,
        datastorm_main_model: str = "gpt-5",
    ):
        self.retriever = retriever
        self.search_top_k = search_top_k
        self.max_thread_num = max_thread_num
        self.enable_followups = enable_followups
        self.generate_graphs = generate_graphs
        self.consolidate_insights = consolidate_insights
        self.langfuse_readonly = langfuse_readonly

        self.conv_simulator = TreeSimulator(
            topic_expert_engine=conv_simulator_lm,
            question_asker_engine=question_asker_lm,
            retriever=retriever,
            max_search_queries_per_turn=max_search_queries_per_turn,
            search_top_k=search_top_k,
            max_tree_depth=max_tree_depth,
            first_level_questions=first_level_questions or [],
            each_level_population_control_num=each_level_population_control_num,
            max_global_insights=max_global_insights,
            db_description=db_description,
            expansion_max_questions=expansion_max_questions,
            enable_followups=enable_followups,
            generate_graphs=generate_graphs,
            consolidate_insights=consolidate_insights,
            langfuse_readonly=langfuse_readonly,
            internet_retriever=internet_retriever,
            thesis_generation_depth=thesis_generation_depth,
            thesis_refinement_interval=thesis_refinement_interval,
            skip_thesis=skip_thesis,
            datastorm_main_model=datastorm_main_model,
        )

    def research(
        self,
        topic: str,
        ground_truth_url: str,
        callback_handler: BaseCallbackHandler,
        return_conversation_log: bool = False,
    ) -> Union[StormInformationTable, Tuple[StormInformationTable, Dict]]:
        """Drive the tree search for `topic` and return the collected information.

        Returns either:
          (information_table, conversation_log_dict, final_selected_res, tree_json)
          when return_conversation_log=True, or
          (information_table, conversation_log_dict, tree_json)
          when return_conversation_log=False.
        """
        callback_handler.on_information_gathering_start()
        # TreeSimulator.__call__ runs its own asyncio event loop via
        # loop.run_until_complete(...), which collides with any outer
        # event loop the caller is already inside (e.g. the entry point's
        # asyncio.run(process_generation(...))). Offload to a worker
        # thread so the inner loop is independent of the outer one.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            prediction = executor.submit(
                self.conv_simulator,
                topic=topic,
                ground_truth_url=ground_truth_url,
                callback_handler=callback_handler,
            ).result()
        # Wrap the single dlg_history into the (persona, history) tuple shape
        # that StormInformationTable still expects. The persona slot is empty
        # since we no longer thread personas through the pipeline.
        conversations = [("", prediction.dlg_history)]
        information_table = StormInformationTable(conversations)
        callback_handler.on_information_gathering_end()

        sim = self.conv_simulator
        tree_json = sim.root.to_json()
        tree_json["thesis"] = sim.current_thesis
        tree_json["research_strategy"] = sim.current_research_strategy
        tree_json["thesis_events"] = sim.thesis_events
        tree_json["failed_queries"] = sim.failed_queries
        tree_json["expansion_mode"] = "global_insight" if sim.use_global_insight_expansion else "per_node"

        log_dict = StormInformationTable.construct_log_dict(conversations)
        if return_conversation_log:
            return information_table, log_dict, sim.final_selected_res, tree_json
        return information_table, log_dict, tree_json
