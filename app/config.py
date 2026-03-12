from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_model: str = "openai/gpt-4o"
    llm_api_key: str = ""
    num_questions: int = 5
    max_evaluator_words: int = 50
    max_answer_length: int = 5000

    model_config = {"env_file": ".env", "extra": "ignore"}

    @model_validator(mode="after")
    def validate_api_key(self):
        if not self.llm_api_key and not self.llm_model.startswith("ollama/"):
            raise ValueError(
                f"LLM_API_KEY is required for model '{self.llm_model}'. "
                "Set it in your .env file. Only ollama/* models work without an API key."
            )
        return self


settings = Settings()
