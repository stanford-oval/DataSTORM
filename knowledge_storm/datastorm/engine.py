import json
import os
from dataclasses import dataclass, field
from typing import Union, List

import dspy

from .modules.callback import BaseCallbackHandler
from .modules.knowledge_curation import StormKnowledgeCurationModule
from .modules.storm_dataclass import StormInformationTable
from ..interface import Engine, LMConfigs, Retriever
from ..utils import FileIOHelper, makeStringRed, truncate_filename


class DataStormLMConfigs(LMConfigs):
    """Configurations for LLMs used by the datatalk research pipeline.

    Only the conversation-simulator and question-asker LMs are used now that
    outline / article / polish stages have been removed in favour of
    `final_report_gen_utils`.
    """

    def __init__(self):
        self.conv_simulator_lm = None
        self.question_asker_lm = None

    def set_conv_simulator_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.conv_simulator_lm = model

    def set_question_asker_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.question_asker_lm = model


@dataclass
class DataStormRunnerArguments:
    """Arguments for controlling the STORM Wiki pipeline."""

    output_dir: str = field(
        metadata={"help": "Output directory for the results."},
    )
    db_description: int = field(
        metadata={
            "help": "A description of the database for Datatalk RM to operate over"
        }
    )
    article_dir_suffix: str = field(
        default="",
        metadata={
            "help": "Optional suffix appended to the per-topic output directory name (e.g. '__YYYYMMDD_HHMMSS')."
        },
    )
    max_search_queries_per_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of search queries to consider in each turn."},
    )
    search_top_k: int = field(
        default=3,
        metadata={"help": "Top k search results to consider for each search query."},
    )
    retrieve_top_k: int = field(
        default=3,
        metadata={"help": "Top k collected references for each section title."},
    )
    max_thread_num: int = field(
        default=10,
        metadata={
            "help": "Maximum number of threads to use. "
            "Consider reducing it if keep getting 'Exceed rate limit' error when calling LM API."
        },
    )
    first_level_questions: List[str] = field(
        default_factory=list,
        metadata={
            "help": "First level questions for the tree search"
        }
    )
    max_tree_depth: int = field(
        default=3,
        metadata={
            "help": "maximum depth of tree search for Datatalk RM"
        }
    )
    additional_evidence: tuple =  field(
        default_factory=tuple,
        metadata={
            "help": "Tuple of evidence from step_1 and additional_evidence in the form of dialogue turns."
        },
    )
    each_level_population_control_num: int = field(
        default=5,
        metadata={
            "help": "In Datatalk RM, on each level, how many nodes to retain"
        }
    )

    max_global_insights: int = field(
        default=30,
        metadata={
            "help": "Maximum number of insights retained in global_insights during tree expansion"
        }
    )

    expansion_max_questions: int = field(
        default=5,
        metadata={
            "help": "Maximum number of questions to generate for node expansion"
        }
    )

    enable_followups: bool = field(
        default=True,
        metadata={
            "help": "If True, enable follow-up questions to clean one level"
        }
    )

    generate_graphs: bool = field(
        default=True,
        metadata={
            "help": "If True, generate graphs"
        }
    )

    consolidate_insights: bool = field(
        default=False,
        metadata={
            "help": "If True, enable final consolidation of insights"
        }
    )
    langfuse_readonly: bool = field(
        default=False,
        metadata={
            "help": "If True, load prompts from Langfuse without recording new generations"
        }
    )
    thesis_generation_depth: int = field(
        default=3,
        metadata={
            "help": "After this many levels, generate a thesis to guide subsequent expansion"
        }
    )
    thesis_refinement_interval: int = field(
        default=2,
        metadata={"help": "Refine thesis every N expansion levels after initial generation"}
    )
    skip_thesis: bool = field(
        default=False,
        metadata={
            "help": "If True, skip thesis generation and refinement. "
                    "Global insights still run and drive next-turn question generation; "
                    "the thesis/research_strategy context is simply omitted from expansion prompts."
        }
    )
    datastorm_main_model: str = field(
        default="gpt-5",
        metadata={
            "help": "Model name to use for internal LLM calls in datastorm (e.g., summarization, reranking, thesis generation)"
        }
    )

class DataStormRunner(Engine):
    """STORM Wiki pipeline runner."""

    def __init__(
        self, args: DataStormRunnerArguments, lm_configs: DataStormLMConfigs, rm,
        internet_rm=None,
    ):
        super().__init__(lm_configs=lm_configs)
        self.args = args
        self.lm_configs = lm_configs

        self.retriever = Retriever(rm=rm, max_thread=self.args.max_thread_num)
        self.internet_retriever = Retriever(rm=internet_rm, max_thread=self.args.max_thread_num) if internet_rm else None
        self.storm_knowledge_curation_module = StormKnowledgeCurationModule(
            retriever=self.retriever,
            conv_simulator_lm=self.lm_configs.conv_simulator_lm,
            question_asker_lm=self.lm_configs.question_asker_lm,
            max_search_queries_per_turn=self.args.max_search_queries_per_turn,
            search_top_k=self.args.search_top_k,
            max_thread_num=self.args.max_thread_num,
            max_tree_depth=self.args.max_tree_depth,
            first_level_questions=self.args.first_level_questions,
            each_level_population_control_num=self.args.each_level_population_control_num,
            max_global_insights=self.args.max_global_insights,
            db_description=self.args.db_description,
            expansion_max_questions=self.args.expansion_max_questions,
            enable_followups=self.args.enable_followups,
            generate_graphs=self.args.generate_graphs,
            consolidate_insights=self.args.consolidate_insights,
            langfuse_readonly=self.args.langfuse_readonly,
            internet_retriever=self.internet_retriever,
            thesis_generation_depth=self.args.thesis_generation_depth,
            thesis_refinement_interval=self.args.thesis_refinement_interval,
            skip_thesis=self.args.skip_thesis,
            datastorm_main_model=self.args.datastorm_main_model,
        )

        self.lm_configs.init_check()
        self.apply_decorators()

    def run_knowledge_curation_module(
        self,
        ground_truth_url: str = "None",
        callback_handler: BaseCallbackHandler = None,
    ) -> StormInformationTable:
        information_table, conversation_log, additional_ev, tree_json = (
            self.storm_knowledge_curation_module.research(
                topic=self.topic,
                ground_truth_url=ground_truth_url,
                callback_handler=callback_handler,
                return_conversation_log=True,
            )
        )

        FileIOHelper.dump_json(
            conversation_log,
            os.path.join(self.article_output_dir, "conversation_log.json"),
        )
        information_table.dump_url_to_info(
            os.path.join(self.article_output_dir, "raw_search_results.json")
        )

        return information_table, additional_ev, tree_json

    def post_run(self):
        """
        Post-run operations, including:
        1. Dumping the run configuration.
        2. Dumping the LLM call history.
        """
        config_log = self.lm_configs.log()
        FileIOHelper.dump_json(
            config_log, os.path.join(self.article_output_dir, "run_config.json")
        )

        llm_call_history = self.lm_configs.collect_and_reset_lm_history()
        with open(
            os.path.join(self.article_output_dir, "llm_call_history.jsonl"), "w"
        ) as f:
            for call in llm_call_history:
                if "kwargs" in call:
                    call.pop(
                        "kwargs"
                    )  # All kwargs are dumped together to run_config.json.
                f.write(json.dumps(call) + "\n")

    def _load_information_table_from_local_fs(self, information_table_local_path):
        assert os.path.exists(information_table_local_path), makeStringRed(
            f"{information_table_local_path} not exists. Please set --do-research argument to prepare the conversation_log.json for this topic."
        )
        return StormInformationTable.from_conversation_log_file(
            information_table_local_path
        )

    def run(
        self,
        topic: str,
        ground_truth_url: str = "",
        callback_handler: BaseCallbackHandler = BaseCallbackHandler(),
    ):
        """Run the knowledge-curation tree search for `topic`.

        Returns a 5-tuple kept for backwards compatibility with existing callers:
            (None, additional_ev, tree_json, article_output_dir, information_table)
        The leading `None` is the legacy "polished article" slot.
        """
        self.topic = topic
        suffix = self.args.article_dir_suffix or ""
        # Ensure the suffix survives truncation so runs are uniquely named like:
        # <topic_slug>__YYYYMMDD_HHMMSS
        base_max_len = max(1, 125 - len(suffix))
        base = truncate_filename(
            topic.replace(" ", "_").replace("/", "_"),
            max_length=base_max_len,
        )
        self.article_dir_name = base + suffix
        self.article_output_dir = os.path.join(
            self.args.output_dir, self.article_dir_name
        )
        os.makedirs(self.article_output_dir, exist_ok=True)

        # Persist the raw user input/topic alongside other artifacts for this run.
        FileIOHelper.write_str(topic + ("\n" if not topic.endswith("\n") else ""), os.path.join(self.article_output_dir, "input.txt"))

        # Route SUQL execution logs into this run's directory so failed /
        # timed-out SUQL queries can be replayed against an improved
        # compiler. Lazy import to avoid coupling the engine package to
        # datatalk_agent at module load time.
        try:
            from knowledge_storm.datatalk_agent.suql_logger import set_log_path as _set_suql_log_path
            _set_suql_log_path(os.path.join(self.article_output_dir, "suql_executions.jsonl"))
        except Exception:
            pass

        information_table, additional_ev, tree_json = self.run_knowledge_curation_module(
            ground_truth_url=ground_truth_url, callback_handler=callback_handler
        )

        if self.args.additional_evidence:
            conversations = information_table.conversations if information_table is not None else []
            # combine information_table with existing evidence
            # Extend information table with evindence from STEP 1
            conversations.extend(self.args.additional_evidence[0].conversations)
            # Append additional evidence from tree search
            conversations.append(("Invesgative journalist: tenacious truth-seeker with a sharp eye for detail and a relentless drive to uncover hidden stories.", self.args.additional_evidence[1]))

            information_table = StormInformationTable(conversations)
            print("🟡🟡🟡🟡🟡 [CHECKPOINT] Already appended new evidence (step_1 & additional_evidence). Length of information_table.conversations after: ", len(information_table.conversations))
            information_table.dump_url_to_info(os.path.join(self.article_output_dir, "url_to_info_with_additional_ev_whole.json"))

        return None, additional_ev, tree_json, self.article_output_dir, information_table
