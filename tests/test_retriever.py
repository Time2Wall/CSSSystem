"""Unit tests for the document retriever."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from app.rag.retriever import DocumentRetriever, RetrievalResult
from app.rag.indexer import DocumentIndexer, Chunk
from app.config import AppConfig


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_retrieval_result_creation(self):
        """Test creating a retrieval result."""
        result = RetrievalResult(
            document_name="test.md",
            content="Test content",
            score=0.1,
            chunk_id="test.md_0"
        )
        assert result.document_name == "test.md"
        assert result.content == "Test content"
        assert result.score == 0.1
        assert result.chunk_id == "test.md_0"

    def test_relevance_percentage_high_similarity(self):
        """Test relevance percentage for high similarity (low distance)."""
        result = RetrievalResult(
            document_name="test.md",
            content="Content",
            score=0.1,  # Low distance = high similarity
            chunk_id="test.md_0"
        )
        assert result.relevance_percentage == 90.0

    def test_relevance_percentage_low_similarity(self):
        """Test relevance percentage for low similarity (high distance)."""
        result = RetrievalResult(
            document_name="test.md",
            content="Content",
            score=0.8,  # High distance = low similarity
            chunk_id="test.md_0"
        )
        assert result.relevance_percentage == 20.0

    def test_relevance_percentage_bounds(self):
        """Test relevance percentage stays within bounds."""
        # Test upper bound
        result1 = RetrievalResult(
            document_name="test.md",
            content="Content",
            score=-0.5,  # Negative distance (shouldn't happen, but test bounds)
            chunk_id="test.md_0"
        )
        assert result1.relevance_percentage == 100.0

        # Test lower bound
        result2 = RetrievalResult(
            document_name="test.md",
            content="Content",
            score=1.5,  # Distance > 1 (shouldn't happen, but test bounds)
            chunk_id="test.md_0"
        )
        assert result2.relevance_percentage == 0.0


class TestDocumentRetriever:
    """Tests for DocumentRetriever class."""

    def test_embed_query(self, test_config: AppConfig, mock_chroma_client):
        """Test query embedding."""
        mock_ollama = Mock()
        expected_embedding = [0.1, 0.2, 0.3]
        mock_ollama.embeddings.return_value = {"embedding": expected_embedding}

        retriever = DocumentRetriever(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        embedding = retriever.embed_query("test query")

        assert embedding == expected_embedding
        mock_ollama.embeddings.assert_called_once_with(
            model=test_config.ollama.embedding_model,
            prompt="test query"
        )

    def test_search_empty_collection(self, test_config: AppConfig, mock_chroma_client):
        """Test search on empty collection."""
        mock_ollama = Mock()
        mock_ollama.embeddings.return_value = {"embedding": [0.1] * 768}

        retriever = DocumentRetriever(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        results = retriever.search("test query")

        assert len(results) == 0

    def test_search_with_results(self, test_config: AppConfig, mock_chroma_client):
        """Test search returns results."""
        mock_ollama = Mock()
        mock_ollama.embeddings.return_value = {"embedding": [0.1] * 768}

        # First, index some documents
        indexer = DocumentIndexer(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        chunks = [
            Chunk(
                document_name="account_opening.md",
                content="How to open a checking account",
                chunk_index=0,
                metadata={"source": "account_opening.md"}
            ),
            Chunk(
                document_name="fees_charges.md",
                content="Monthly maintenance fees and overdraft charges",
                chunk_index=0,
                metadata={"source": "fees_charges.md"}
            ),
        ]
        indexer.index_chunks(chunks)

        # Now search
        retriever = DocumentRetriever(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        results = retriever.search("open account", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.document_name in ["account_opening.md", "fees_charges.md"] for r in results)

    def test_search_top_k_limit(self, test_config: AppConfig, mock_chroma_client):
        """Test search respects top_k limit."""
        mock_ollama = Mock()
        mock_ollama.embeddings.return_value = {"embedding": [0.1] * 768}

        # Index multiple chunks
        indexer = DocumentIndexer(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        chunks = [
            Chunk(document_name=f"doc{i}.md", content=f"Content {i}", chunk_index=0, metadata={"source": f"doc{i}.md"})
            for i in range(5)
        ]
        indexer.index_chunks(chunks)

        retriever = DocumentRetriever(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        results = retriever.search("content", top_k=2)

        assert len(results) == 2

    def test_get_document_content(self, sample_knowledge_base: Path, test_config: AppConfig, mock_chroma_client):
        """Test getting full document content."""
        with patch('ollama.Client'):
            retriever = DocumentRetriever(
                config=test_config,
                chroma_client=mock_chroma_client
            )

            content = retriever.get_document_content(
                "account_opening.md",
                str(sample_knowledge_base)
            )

            assert content is not None
            assert "Checking Account" in content

    def test_get_document_content_not_found(self, sample_knowledge_base: Path, test_config: AppConfig, mock_chroma_client):
        """Test getting content of nonexistent document."""
        with patch('ollama.Client'):
            retriever = DocumentRetriever(
                config=test_config,
                chroma_client=mock_chroma_client
            )

            content = retriever.get_document_content(
                "nonexistent.md",
                str(sample_knowledge_base)
            )

            assert content is None

    def test_get_document_names_empty(self, test_config: AppConfig, mock_chroma_client):
        """Test getting document names from empty collection."""
        with patch('ollama.Client'):
            retriever = DocumentRetriever(
                config=test_config,
                chroma_client=mock_chroma_client
            )

            names = retriever.get_document_names()

            assert len(names) == 0

    def test_get_document_names_with_documents(self, test_config: AppConfig, mock_chroma_client):
        """Test getting document names after indexing."""
        mock_ollama = Mock()
        mock_ollama.embeddings.return_value = {"embedding": [0.1] * 768}

        indexer = DocumentIndexer(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        chunks = [
            Chunk(document_name="doc1.md", content="Content 1", chunk_index=0, metadata={"source": "doc1.md"}),
            Chunk(document_name="doc1.md", content="Content 1b", chunk_index=1, metadata={"source": "doc1.md"}),
            Chunk(document_name="doc2.md", content="Content 2", chunk_index=0, metadata={"source": "doc2.md"}),
        ]
        indexer.index_chunks(chunks)

        retriever = DocumentRetriever(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        names = retriever.get_document_names()

        assert len(names) == 2
        assert "doc1.md" in names
        assert "doc2.md" in names

    def test_search_with_context(self, test_config: AppConfig, mock_chroma_client):
        """Test search with combined context."""
        mock_ollama = Mock()
        mock_ollama.embeddings.return_value = {"embedding": [0.1] * 768}

        # Index documents
        indexer = DocumentIndexer(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        chunks = [
            Chunk(document_name="doc1.md", content="Account opening info", chunk_index=0, metadata={"source": "doc1.md"}),
            Chunk(document_name="doc2.md", content="Fee information", chunk_index=0, metadata={"source": "doc2.md"}),
        ]
        indexer.index_chunks(chunks)

        retriever = DocumentRetriever(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        results, context = retriever.search_with_context("information", top_k=2)

        assert len(results) == 2
        assert "[Source 1:" in context
        assert "[Source 2:" in context
        assert "---" in context

    def test_search_with_context_empty(self, test_config: AppConfig, mock_chroma_client):
        """Test search with context on empty collection."""
        mock_ollama = Mock()
        mock_ollama.embeddings.return_value = {"embedding": [0.1] * 768}

        retriever = DocumentRetriever(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        results, context = retriever.search_with_context("test")

        assert len(results) == 0
        assert context == ""
