"""Pytest fixtures for testing the Customer Service Support System."""

import asyncio
import tempfile
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock, MagicMock
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import AppConfig, OllamaConfig, ChromaConfig, DatabaseConfig, RAGConfig


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config(tmp_path: Path) -> AppConfig:
    """Create test configuration with temporary directories."""
    return AppConfig(
        ollama=OllamaConfig(
            host="http://localhost:11434",
            llm_model="llama3.2:3b",
            embedding_model="nomic-embed-text",
            timeout=30
        ),
        chroma=ChromaConfig(
            collection_name="test_banking_knowledge",
            persist_directory=str(tmp_path / "chroma")
        ),
        database=DatabaseConfig(
            url=f"sqlite:///{tmp_path / 'test.db'}",
            echo=False
        ),
        rag=RAGConfig(
            chunk_size=200,
            chunk_overlap=20,
            top_k=2
        ),
        debug=True
    )


@pytest.fixture
def mock_ollama_client() -> Mock:
    """Create a mock Ollama client for testing."""
    client = Mock()

    # Mock chat method (for LLM calls)
    async def mock_chat(model, messages, **kwargs):
        return {
            "message": {
                "content": "This is a mock response from the LLM."
            }
        }

    client.chat = AsyncMock(side_effect=mock_chat)

    # Mock embeddings method
    async def mock_embeddings(model, prompt, **kwargs):
        # Return a fake embedding vector (768 dimensions for nomic-embed-text)
        import hashlib
        # Generate deterministic embeddings based on prompt content
        hash_val = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        return {
            "embedding": [(hash_val >> i) % 1000 / 1000.0 for i in range(768)]
        }

    client.embeddings = AsyncMock(side_effect=mock_embeddings)

    return client


@pytest.fixture
def mock_chroma_client(tmp_path: Path) -> MagicMock:
    """Create a mock ChromaDB client for testing."""
    import chromadb

    # Use ephemeral client for testing
    client = chromadb.Client()
    return client


@pytest.fixture
def sample_documents() -> list[dict]:
    """Sample documents for testing."""
    return [
        {
            "name": "account_opening.md",
            "content": """# Account Opening Policies

## Checking Account
To open a checking account, you need:
- Valid government-issued ID
- Proof of address (utility bill or bank statement)
- Minimum deposit of $25

## Savings Account
Requirements for savings account:
- Valid ID
- Social Security Number
- Minimum deposit of $100
"""
        },
        {
            "name": "fees_charges.md",
            "content": """# Fees and Charges

## Monthly Maintenance Fees
- Basic Checking: $5/month (waived with $500 minimum balance)
- Premium Checking: $15/month (waived with $2,500 minimum balance)

## Overdraft Fees
- Standard overdraft fee: $35 per transaction
- Overdraft protection transfer: $10 per transfer

## Refund Policy
Fees may be refunded in the following cases:
- Bank error
- First-time courtesy refund
- Account in good standing for 12+ months
"""
        }
    ]


@pytest.fixture
def sample_knowledge_base(tmp_path: Path, sample_documents: list[dict]) -> Path:
    """Create a temporary knowledge base directory with sample documents."""
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()

    for doc in sample_documents:
        (kb_dir / doc["name"]).write_text(doc["content"])

    return kb_dir


@pytest.fixture
def test_db_session(test_config: AppConfig):
    """Create a test database session."""
    from app.database.models import Base

    engine = create_engine(test_config.database.url, echo=False)
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    yield session

    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture
async def async_client(test_config: AppConfig):
    """Create async test client for FastAPI."""
    from httpx import AsyncClient, ASGITransport
    from app.main import create_app

    app = create_app(test_config)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client


# Marker for integration tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
