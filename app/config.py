"""Configuration settings for the Customer Service Support System."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"


@dataclass
class OllamaConfig:
    """Ollama LLM configuration."""
    host: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text"
    timeout: int = 120


@dataclass
class ChromaConfig:
    """ChromaDB configuration."""
    collection_name: str = "banking_knowledge"
    persist_directory: Optional[str] = None

    def __post_init__(self):
        if self.persist_directory is None:
            self.persist_directory = str(DATA_DIR / "chroma")


@dataclass
class DatabaseConfig:
    """SQLite database configuration."""
    url: str = field(default_factory=lambda: f"sqlite:///{DATA_DIR / 'css_system.db'}")
    echo: bool = False


@dataclass
class RAGConfig:
    """RAG system configuration."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3


@dataclass
class AppConfig:
    """Main application configuration."""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    # Confidence thresholds
    high_confidence_threshold: int = 70
    low_confidence_threshold: int = 40

    # API settings
    api_prefix: str = "/api"
    debug: bool = False


def get_config() -> AppConfig:
    """Get application configuration, optionally loading from environment."""
    config = AppConfig()

    # Override from environment variables if present
    if os.getenv("OLLAMA_HOST"):
        config.ollama.host = os.getenv("OLLAMA_HOST")
    if os.getenv("OLLAMA_LLM_MODEL"):
        config.ollama.llm_model = os.getenv("OLLAMA_LLM_MODEL")
    if os.getenv("OLLAMA_EMBEDDING_MODEL"):
        config.ollama.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL")
    if os.getenv("DATABASE_URL"):
        config.database.url = os.getenv("DATABASE_URL")
    if os.getenv("DEBUG"):
        config.debug = os.getenv("DEBUG", "").lower() in ("true", "1", "yes")

    return config


# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
