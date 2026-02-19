import litellm
from app.config import settings


async def complete(messages: list[dict], temperature: float = 0.7) -> str:
    """Call LLM and return the full response text."""
    response = await litellm.acompletion(
        model=settings.llm_model,
        messages=messages,
        api_key=settings.llm_api_key,
        temperature=temperature,
    )
    return response.choices[0].message.content


async def complete_json(messages: list[dict], temperature: float = 0.3) -> str:
    """Call LLM with JSON response format."""
    response = await litellm.acompletion(
        model=settings.llm_model,
        messages=messages,
        api_key=settings.llm_api_key,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content
