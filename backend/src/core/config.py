from pydantic_settings import BaseSettings, SettingsConfigDict
import dotenv

dotenv.load_dotenv()

class Config(BaseSettings):
    EMBEDDING_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    AGENTIC_MODEL: str = "z-ai/glm-5.2"
    GEMINI_API_KEY: str | None = None
    GROQ_API_KEY:str
    QDRANT_URL: str = "http://qdrant:6333"
    AI_API_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


config = Config()