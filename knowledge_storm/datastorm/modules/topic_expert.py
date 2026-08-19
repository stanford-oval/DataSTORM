from typing import List, Optional, Union

import dspy

from ...interface import Information, Retriever


class TopicExpert(dspy.Module):
    """Route a question to the configured retriever and surface the result.

    The datatalk RMs (DatatalkRM, DecompositionAgentRM) already return
    structured summaries in their snippets, so this module is a thin wrapper:
    it picks the right retriever (database vs. internet) based on the
    `designation` key, threads through `conversation_history` for follow-ups,
    and concatenates snippets into the `answer` field expected by callers.
    """

    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        max_search_queries: int,
        search_top_k: int,
        retriever: Retriever,
        internet_retriever: Optional[Retriever] = None,
    ):
        super().__init__()
        self.retriever = retriever
        self.internet_retriever = internet_retriever
        self.engine = engine
        self.max_search_queries = max_search_queries
        self.search_top_k = search_top_k

    def forward(
        self,
        topic: str,
        question: Union[str, dict],
        ground_truth_url: str,
        conversation_history: Optional[str] = None
    ):
        designation = "question"
        # some callers pass in a dict with a designation key
        if isinstance(question, dict):
            designation = question.get("designation", "question")
            question = question["question"]

        queries = [question]
        to_retrieve = list(set(queries))
        print(f"retrieving {to_retrieve} with designation {designation}")
        if designation == "internet" and self.internet_retriever is not None:
            searched_results: List[Information] = self.internet_retriever.retrieve(
                to_retrieve, exclude_urls=[ground_truth_url]
            )
            for r in searched_results:
                r.meta["source_type"] = "internet"
        elif conversation_history is None:
            searched_results: List[Information] = self.retriever.retrieve(
                to_retrieve, exclude_urls=[ground_truth_url], designation=designation
            )
            for r in searched_results:
                r.meta.setdefault("source_type", "database")
        else:
            searched_results: List[Information] = self.retriever.retrieve(
                to_retrieve, exclude_urls=[ground_truth_url], conversation_history=conversation_history, designation=designation
            )
            for r in searched_results:
                r.meta.setdefault("source_type", "database")

        if searched_results:
            answer = ""
            for n, r in enumerate(searched_results):
                answer += "\n".join(f"[{n + 1}]: {s}" for s in r.snippets[:1])
                answer += "\n\n"
        else:
            # When no information is found, the expert shouldn't hallucinate.
            answer = "Sorry, I cannot find information for this question. Please ask another question."

        return dspy.Prediction(
            queries=queries, searched_results=searched_results, answer=answer
        )
