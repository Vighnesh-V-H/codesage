from pydantic_settings import BaseSettings, SettingsConfigDict
import dotenv

dotenv.load_dotenv()

class Config(BaseSettings):
    EMBEDDING_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    AGENTIC_MODEL: str = "z-ai/glm-5.2"
    GEMINI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    EMBEDDING_MODEL: str = "nvidia/llama-nemotron-embed-1b-v2"
    QDRANT_URL: str = "http://qdrant:6333"
    AI_API_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/1"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


config = Config()