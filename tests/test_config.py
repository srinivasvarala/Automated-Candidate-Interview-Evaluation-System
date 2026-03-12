import pytest


def test_settings_loads_from_env():
    """Settings should load from environment variables."""
    from app.config import settings
    assert settings.llm_model == "openai/gpt-4o"
    assert settings.llm_api_key == "test-key-for-testing"
    assert settings.num_questions == 3


def test_settings_has_defaults():
    """Settings should have sensible defaults."""
    from app.config import settings
    assert settings.max_evaluator_words == 50
    assert settings.max_answer_length == 5000


def test_settings_rejects_empty_api_key_for_non_ollama():
    """Settings should raise if API key is empty for non-ollama models."""
    from app.config import Settings
    with pytest.raises(ValueError, match="LLM_API_KEY is required"):
        Settings(llm_model="openai/gpt-4o", llm_api_key="")


def test_settings_allows_empty_api_key_for_ollama():
    """Settings should allow empty API key for ollama models."""
    from app.config import Settings
    s = Settings(llm_model="ollama/llama3", llm_api_key="")
    assert s.llm_api_key == ""
