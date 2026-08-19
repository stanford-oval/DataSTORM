import dspy
from typing import List, Union

from .collaborative_storm_utils import extract_and_remove_citations
from ...dataclass import ConversationTurn


class AskQuestionWithPersona(dspy.Signature):
    """You are an experienced Wikipedia writer and want to edit a specific page. Besides your identity as a Wikipedia writer, you have specific focus when researching the topic.
    Now, you are chatting with an expert to get information. Ask good questions to get more useful information.
    When you have no more question to ask, say "Thank you so much for your help!" to end the conversation.
    Please only ask a question at a time and don't ask what you have asked before. Your questions should be related to the topic you want to write.
    """

    topic = dspy.InputField(prefix="Topic you want to write: ", format=str)
    persona = dspy.InputField(
        prefix="Your persona besides being a Wikipedia writer: ", format=str
    )
    conv = dspy.InputField(prefix="Conversation history:\n", format=str)
    question = dspy.OutputField(format=str)


class GenSimulatedUserUtterance(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.engine = engine
        self.ask_qeustion = dspy.Predict(AskQuestionWithPersona)

    def gen_conv_history_string(self, conversation_turns: List[ConversationTurn]):
        conv_history = []
        total_turns = len(conversation_turns)

        for i, turn in enumerate(conversation_turns):
            utterance, _ = extract_and_remove_citations(turn.utterance)
            if i >= total_turns - 4:
                conv_history.append(f"{turn.role}: {utterance}")
            else:
                if turn.claim_to_make:
                    conv_history.append(f"{turn.role}: {turn.claim_to_make}")
                else:
                    conv_history.append(f"{turn.role}: {utterance}")

        return "\n".join(conv_history)

    def forward(self, topic: str, intent: str, conv_history: List[ConversationTurn]):
        conv_history_string = self.gen_conv_history_string(conv_history)
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            return self.ask_qeustion(
                topic=topic,
                persona=f"researcher with interest in {intent}",
                conv=conv_history_string,
            ).question
