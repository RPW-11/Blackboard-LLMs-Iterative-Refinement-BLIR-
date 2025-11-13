from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path


class LlmConfig(BaseSettings):
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    
    class Config:
        env_file = Path(__file__).parent.parent.parent / ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False