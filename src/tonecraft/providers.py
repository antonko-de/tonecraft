"""Multi-provider instructor client factory (Anthropic, OpenAI, Ollama)."""

import logging

import instructor

# Lazy optional imports — set to None if not installed so the module loads
# cleanly regardless of which provider extras the user has installed.
# Callers get a clear error at create_client() time, not at import time.
try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_SUPPORTED = ("anthropic", "openai", "ollama")


def create_client(provider: str, model: str, base_url: str | None = None):
    """Return an instructor-patched client for the given provider.

    Args:
        provider: One of "anthropic", "openai", "ollama".
        model:    Model name (used by the caller when making completions).
        base_url: Optional base URL override. Required for Ollama if not
                  using the default localhost address.

    Returns:
        An instructor.Instructor-patched client ready for structured output.
    """
    if provider not in _SUPPORTED:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: {', '.join(_SUPPORTED)}."
        )

    if provider == "anthropic":
        if anthropic is None:
            raise ImportError(
                "Package 'anthropic' is not installed. "
                "Install it with: pip install tonecraft[anthropic]"
            )
        logger.debug("Creating Anthropic instructor client (model=%s)", model)
        return instructor.from_anthropic(anthropic.Anthropic())

    if provider == "openai":
        if openai is None:
            raise ImportError(
                "Package 'openai' is not installed. "
                "Install it with: pip install tonecraft[openai]"
            )
        logger.debug("Creating OpenAI instructor client (model=%s)", model)
        return instructor.from_openai(openai.OpenAI())

    # ollama — OpenAI-compatible API endpoint
    if openai is None:
        raise ImportError(
            "Package 'openai' is not installed. "
            "Install it with: pip install tonecraft[openai]"
        )
    effective_url = base_url or "http://localhost:11434/v1"
    logger.debug("Creating Ollama instructor client (model=%s, base_url=%s)", model, effective_url)
    return instructor.from_openai(openai.OpenAI(base_url=effective_url, api_key="ollama"))
