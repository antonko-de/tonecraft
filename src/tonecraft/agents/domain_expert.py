import logging

from tonecraft.schemas import AgentBrief, ContextDocument, DistributionSchema, ProjectConfig

logger = logging.getLogger(__name__)


def analyze(doc: ContextDocument, config: ProjectConfig, client) -> tuple[DistributionSchema, AgentBrief, AgentBrief]:
    topics_list = ", ".join(doc.topics)

    logger.debug("Requesting distribution schema for domain: %s", doc.domain)
    distribution = client.chat.completions.create(
        model=config.generation.model_expert,
        response_model=DistributionSchema,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"You are a domain expert analyzing the '{doc.domain}' domain.\n"
                f"Topics: {topics_list}\n"
                f"Target SLM: {config.target.slm} (context window: {config.target.slm_context_window})\n"
                f"Target max pairs: {config.generation.max_pairs}\n\n"
                "Create a distribution schema covering ALL listed topics with weights, "
                "target counts, a recommended total pair count, and a rationale for the data sizing."
            ),
        }],
    )

    topic_summary = ", ".join(t.topic for t in distribution.topics)

    logger.debug("Requesting questioner brief")
    q_brief = client.chat.completions.create(
        model=config.generation.model_expert,
        response_model=AgentBrief,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"Create an agent brief for the questioner role: '{doc.questioner_role}'.\n"
                f"Tone guidelines: {doc.tone_guidelines}\n"
                f"Topics covered: {topic_summary}\n"
                "The role field must be 'questioner'."
            ),
        }],
    )

    logger.debug("Requesting responder brief")
    r_brief = client.chat.completions.create(
        model=config.generation.model_expert,
        response_model=AgentBrief,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"Create an agent brief for the responder role: '{doc.responder_role}'.\n"
                f"Tone guidelines: {doc.tone_guidelines}\n"
                f"Topics covered: {topic_summary}\n"
                "The role field must be 'responder'."
            ),
        }],
    )

    return distribution, q_brief, r_brief
