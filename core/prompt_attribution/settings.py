"""Configuration settings for the Prompt Attribution toolkit."""

import os
from typing import Optional

from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseModel):
    """Application settings validated with Pydantic."""
    
    # API Configuration
    openai_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("OPENAI_API_KEY", ""))
    )
    
    # Model Configuration  
    completion_model: str = Field(
        default=os.getenv("COMPLETION_MODEL", "gpt-4o")
    )
    embedding_model: str = Field(
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    
    # Cost Guardrails
    max_cost_per_run: float = Field(
        default=float(os.getenv("MAX_COST_PER_RUN", "0.15"))
    )
    
    # Performance Settings
    max_concurrent_requests: int = Field(
        default=int(os.getenv("MAX_CONCURRENT_REQUESTS", "20"))
    )
    early_stop_threshold: float = Field(
        default=float(os.getenv("EARLY_STOP_THRESHOLD", "0.85"))
    )
    
    # Cache Settings
    enable_cache: bool = Field(
        default=os.getenv("ENABLE_CACHE", "true").lower() == "true"
    )
    cache_dir: Optional[str] = Field(
        default=os.getenv("CACHE_DIR", ".prompt_attribution_cache")
    )


# Create a global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Return the global settings instance."""
    return settings 