import logging

import litellm
from app.config import settings

logger = logging.getLogger(__name__)


def _extract_content(response) -> str:
    """Extract text content from LLM response, handling provider differences."""
    content = response.choices[0].message.content
    logger.debug("Raw LLM content type=%s, value=%s", type(content).__name__, repr(content)[:200] if content else "None")
    if content is None:
        # Anthropic can return content as None when using content blocks
        # Fall back to extracting from the raw response
        try:
            blocks = response.choices[0].message.tool_calls or []
            if not blocks:
                # Try content blocks from raw Anthropic response
                raw = response._raw_response
                if hasattr(raw, "content") and isinstance(raw.content, list):
                    texts = [b.text for b in raw.content if hasattr(b, "text")]
                    return "\n".join(texts) if texts else ""
        except (AttributeError, IndexError):
            pass
        return ""
    return content


async def complete(messages: list[dict], temperature: float = 0.7) -> str:
    """Call LLM and return the full response text."""
    logger.debug("LLM call: model=%s, messages=%d", settings.llm_model, len(messages))
    response = await litellm.acompletion(
        model=settings.llm_model,
        messages=messages,
        api_key=settings.llm_api_key,
        temperature=temperature,
        num_retries=2,
        request_timeout=60,
    )
    result = _extract_content(response)
    if not result:
        logger.warning("LLM returned empty content for complete() call")
    return result


async def complete_json(messages: list[dict], temperature: float = 0.3) -> str:
    """Call LLM expecting JSON. Extracts JSON from response even without response_format support."""
    logger.debug("LLM JSON call: model=%s, messages=%d", settings.llm_model, len(messages))

    # Anthropic via LiteLLM may not support response_format, so we don't rely on it.
    # Instead, we just call normally and extract JSON from the text.
    kwargs = {
        "model": settings.llm_model,
        "messages": messages,
        "api_key": settings.llm_api_key,
        "temperature": temperature,
        "num_retries": 2,
        "request_timeout": 60,
    }

    # Only add response_format for providers that support it (OpenAI-compatible)
    model = settings.llm_model.lower()
    if not model.startswith("anthropic/") and not model.startswith("claude"):
        kwargs["response_format"] = {"type": "json_object"}

    response = await litellm.acompletion(**kwargs)
    result = _extract_content(response)

    if not result:
        logger.warning("LLM returned empty content for complete_json() call")
        return "{}"

    # If the response contains JSON wrapped in markdown code blocks, extract it
    stripped = result.strip()
    if stripped.startswith("```"):
        # Remove ```json or ``` prefix and trailing ```
        lines = stripped.split("\n")
        # Find start and end of code block
        start = 1  # skip first ``` line
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        result = "\n".join(lines[start:end]).strip()

    logger.debug("LLM JSON response: %s", result[:200])
    return result
