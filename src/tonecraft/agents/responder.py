import logging

from tonecraft.schemas import AgentBrief, QAPair

logger = logging.getLogger(__name__)


def generate_response(brief: AgentBrief, question: str, topic: str, client, model: str) -> QAPair:
    logger.debug("Generating response for topic: %s", topic)
    return client.chat.completions.create(
        model=model,
        response_model=QAPair,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"You are: {brief.persona}\n"
                f"Directives: {', '.join(brief.directives)}\n"
                f"Question: {question}\n"
                f"Topic: {topic}\n\n"
                "Generate a QAPair with the question, your answer, the topic, "
                "and a confidence score between 0.0 and 1.0."
            ),
        }],
    )
