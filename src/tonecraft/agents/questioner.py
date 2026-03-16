import logging

from pydantic import BaseModel

from tonecraft.schemas import AgentBrief

logger = logging.getLogger(__name__)


class _Question(BaseModel):
    text: str


def generate_question(brief: AgentBrief, topic: str, client, model: str) -> str:
    logger.debug("Generating question for topic: %s", topic)
    result = client.chat.completions.create(
        model=model,
        response_model=_Question,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                f"You are: {brief.persona}\n"
                f"Directives: {', '.join(brief.directives)}\n"
                f"Topic: {topic}\n"
                f"Topic focus: {brief.topic_focus}\n\n"
                "Generate one natural question."
            ),
        }],
    )
    return result.text
