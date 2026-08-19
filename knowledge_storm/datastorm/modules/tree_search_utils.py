import numpy as np
import math
from typing import List
from ...interface import Information
from typing import Union, List, Tuple, Optional, Dict
from .storm_dataclass import DialogueTurn
import uuid



class TreeNode():
    def __init__(
        self,
        agent_utterance: str = None,
        user_utterance: str = None,
        search_queries: Optional[List[str]] = None,
        conversation_history: Optional[str] = None,
        search_results: Optional[List[Union[Information, Dict]]] = None,
        parent = None,
        children = None,
        depth = 0,
        result_count = -1,
        summary: Optional[str] = None,
        interesting_rerank_score: Optional[float] = None,
        is_final_selected: Optional[bool] = None,
        is_follow_up: bool = False,
        node_id: Optional[str] = None,
    ):
        self.node_id = node_id if node_id else str(uuid.uuid4())
        self.dlg_turn = DialogueTurn(
            agent_utterance=agent_utterance,
            user_utterance=user_utterance,
            search_queries=search_queries,
            search_results=search_results,
            summary=summary
        )
        self.conversation_history = conversation_history
        self.parent = parent
        self.children = [] if children is None else children
        self.depth = depth
        self.expanded = False
        self.result_count = result_count
        self.interesting_rerank_score = interesting_rerank_score
        self.is_final_selected = is_final_selected
        self.is_follow_up = is_follow_up
        
    
    def reconstruct_dialog_history(
        self,
        selection_criteria: callable = None
    ):
        """
            This function goes back and constructs a dialog history up to this point
            
            NOTE: this is the high-level conversation for use with STORM. For low-level conversation
            with actions and results, that is recorded within each node's conversation_history
        """
        dlg_history: List[DialogueTurn] = []
        cur_node = self
        while cur_node.parent is not None:
            if selection_criteria is None or selection_criteria(cur_node):
                dlg_history = [cur_node.dlg_turn] + dlg_history
            cur_node = cur_node.parent
            
        assert(cur_node.depth == 0)
        
        return dlg_history
    
    def construct_all_dialog_paths(
        self,
        include_self=False,
        append_node=False,
        ignore_none_follow_ups=True
    ):
        """
            Traverse the tree in a BFS manner and collect all dialogue turns.
            
            Use this function to serialize the tree to feed into STORM.
        """
        from collections import deque

        queue = deque([self] if include_self else self.children)
        res: List[DialogueTurn] = []

        while queue:
            node = queue.popleft()
            
            if not ignore_none_follow_ups or node.is_follow_up:
                if append_node:
                    res.append(node)
                else:
                    res.append(node.dlg_turn)
            
            queue.extend(node.children)

        return res
    
    def get_node_by_id(self, node_id: str):
        """
            Get a node by its node_id
        """
        if self.node_id == node_id:
            return self
        for child in self.children:
            res = child.get_node_by_id(node_id)
            if res:
                return res
        return None
        
    def to_json(self):
        """Serialize the tree node's child nodes and their descendants to JSON format"""
        return {
            'dlg_turn': {
                'agent_utterance': self.dlg_turn.agent_utterance,
                'user_utterance': self.dlg_turn.user_utterance,
                'search_queries': self.dlg_turn.search_queries,
                'summary': self.dlg_turn.summary,
                'search_results': [
                    {
                        'url': result.url if hasattr(result, 'url') else None,
                        'snippets': result.snippets if hasattr(result, 'snippets') else [],
                        'title': result.title if hasattr(result, 'title') else None,
                        'description': result.description if hasattr(result, 'description') else None,
                        'meta': result.meta if hasattr(result, 'meta') else {},
                        'citation_uuid': result.citation_uuid if hasattr(result, 'citation_uuid') else -1,
                        'conversation_history': result.conversation_history if hasattr(result, 'conversation_history') else None,
                    } for result in (self.dlg_turn.search_results or [])
                ],
            },
            'depth': self.depth,
            'children': [child.to_json() for child in self.children],
            "interesting_rerank_score": self.interesting_rerank_score,
            'is_final_selected': self.is_final_selected,
            "is_follow_up": self.is_follow_up,
            "node_id": self.node_id
        }

    @classmethod 
    def from_json(cls, json_data):
        """Deserialize JSON format to reconstruct the tree node's child nodes and their descendants."""
        dlg_turn_data = json_data.get('dlg_turn', {})
        dlg_turn = DialogueTurn(
            agent_utterance=dlg_turn_data.get('agent_utterance'),
            user_utterance=dlg_turn_data.get('user_utterance'),
            search_queries=dlg_turn_data.get('search_queries', []),
            summary=dlg_turn_data.get('summary', None),
            search_results=[
                Information(
                    url=result.get('url'),
                    snippets=result.get('snippets', []),
                    title=result.get('title'),
                    description=result.get('description'),
                    meta=result.get('meta', {}),
                    citation_uuid=result.get('citation_uuid', -1),
                    conversation_history=result.get('conversation_history'),
                ) for result in dlg_turn_data.get('search_results', [])
            ]
        )
        
        children = [cls.from_json(child_json) for child_json in json_data.get('children', [])]
        
        return cls(
            dlg_turn=dlg_turn,
            depth=json_data.get('depth', 0),
            interesting_rerank_score=json_data.get('interesting_rerank_score', None),
            is_final_selected=json_data.get('is_final_selected', None),
            is_follow_up=json_data.get("is_follow_up", False),
            children=children,
            node_id=json_data.get("node_id"),
        )
        
    def get_lowest_level_nodes(node, exclude_no_results = True):
        """
            Returns the nodes that are at the lowest level of the tree (those that do not have children)
        """
        lowest_nodes = []
        
        def traverse(current_node: TreeNode):
            if not current_node.children:
                if not exclude_no_results or current_node.result_count > 0:
                    lowest_nodes.append(current_node)
            else:
                for child in current_node.children:
                    traverse(child)

        traverse(node)
        return lowest_nodes

    def regenerate_node_id(self):
        """
        Generate a new unique node_id for this node.
        """
        self.node_id = str(uuid.uuid4())
        
    def find_nodes_on_level(self, level: int):
        """
            Find all nodes on a given level.
        """
        if level <= 0:
            return []
            
        nodes_on_level = []
        
        # Use BFS to find nodes at the specified level
        from collections import deque
        queue = deque([(self, 0)])  # (node, depth)
        
        while queue:
            node, depth = queue.popleft()
            
            if depth == level:
                nodes_on_level.append(node)
            elif depth < level:
                for child in node.children:
                    queue.append((child, depth + 1))
            
            # If we've processed all nodes up to our target level,
            # and there are no more nodes at our target level in the queue,
            # we can stop searching
            if depth >= level and all(d > level for _, d in queue):
                break
        
        return nodes_on_level


def llm_output_to_yes_probability(llm_output: tuple) -> float:
    """
    Converts the LLM output to a probability of the answer being 'yes'.

    Args:
        llm_output (tuple): The output from the LLM, including log probabilities.

    Returns:
        float: The probability of the answer being 'yes'.
    """
    T = 1  # temperature scaling

    logprob_outputs = llm_output[1][0]["top_logprobs"]

    # Filter tokens containing "yes" or "no" (excluding "not")
    filtered_logprobs = [
        lpo
        for lpo in logprob_outputs
        if "yes" in lpo["token"].lower()
        or ("no" in lpo["token"].lower() and "not" not in lpo["token"].lower())
    ]

    # Extract logprobs and apply temperature scaling
    logprobs = np.array([lpo["logprob"] for lpo in filtered_logprobs])
    scaled_probs = np.exp(logprobs / T)
    normalized_probs = scaled_probs / np.sum(scaled_probs)

    # Calculate yes and no probabilities
    yes_prob = sum(
        prob
        for prob, lpo in zip(normalized_probs, filtered_logprobs)
        if "yes" in lpo["token"].lower()
    )
    no_prob = sum(
        prob
        for prob, lpo in zip(normalized_probs, filtered_logprobs)
        if "no" in lpo["token"].lower()
    )

    # Ensure the probabilities sum to 1
    assert math.isclose(yes_prob + no_prob, 1, rel_tol=1e-3)

    return yes_prob