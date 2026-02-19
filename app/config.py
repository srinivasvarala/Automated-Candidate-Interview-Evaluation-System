from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_model: str = "openai/gpt-4o"
    llm_api_key: str = ""
    num_questions: int = 3
    max_evaluator_words: int = 50

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
