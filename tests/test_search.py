"""Unit tests for the Search Agent."""

import pytest
import json
from unittest.mock import Mock, patch

from app.agents.search import SearchAgent, SearchResult
from app.rag.retriever import DocumentRetriever, RetrievalResult
from app.config import AppConfig


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_result_creation(self):
        """Test creating a search result."""
        result = SearchResult(
            query="test query",
            answer="Test answer",
            source_document="test.md",
            relevant_passages=[]
        )
        assert result.query == "test query"
        assert result.answer == "Test answer"
        assert result.source_document == "test.md"
        assert result.relevant_passages == []

    def test_result_with_passages(self):
        """Test search result with passages."""
        passages = [
            RetrievalResult(
                document_name="doc1.md",
                content="Passage 1",
                score=0.1,
                chunk_id="doc1.md_0"
            )
        ]
        result = SearchResult(
            query="test",
            answer="answer",
            source_document="doc1.md",
            relevant_passages=passages
        )
        assert len(result.relevant_passages) == 1
        assert result.relevant_passages[0].content == "Passage 1"


class TestSearchAgent:
    """Tests for SearchAgent class."""

    @pytest.fixture
    def mock_retriever_with_results(self, test_config: AppConfig):
        """Create mock retriever with configurable results."""
        def _create_mock(results: list[RetrievalResult], context: str = ""):
            mock = Mock(spec=DocumentRetriever)
            mock.search_with_context.return_value = (results, context)
            return mock
        return _create_mock

    @pytest.fixture
    def mock_ollama_with_response(self):
        """Create mock Ollama client with configurable response."""
        def _create_mock(response_json: dict):
            mock = Mock()
            mock.chat.return_value = {
                "message": {
                    "content": json.dumps(response_json)
                }
            }
            return mock
        return _create_mock

    @pytest.mark.asyncio
    async def test_search_basic(self, test_config: AppConfig, mock_retriever_with_results, mock_ollama_with_response):
        """Test basic search functionality."""
        passages = [
            RetrievalResult(
                document_name="account_opening.md",
                content="To open a checking account, you need valid ID and minimum deposit.",
                score=0.1,
                chunk_id="account_opening.md_0"
            )
        ]
        context = "[Source 1: account_opening.md]\nTo open a checking account, you need valid ID and minimum deposit."

        mock_retriever = mock_retriever_with_results(passages, context)
        mock_ollama = mock_ollama_with_response({
            "answer": "To open a checking account, you need a valid ID and a minimum deposit.",
            "primary_source": "account_opening.md"
        })

        agent = SearchAgent(
            config=test_config,
            ollama_client=mock_ollama,
            retriever=mock_retriever
        )

        result = await agent.search("how to open checking account")

        assert "checking account" in result.answer.lower() or "valid ID" in result.answer
        assert result.source_document == "account_opening.md"
        assert len(result.relevant_passages) == 1

    @pytest.mark.asyncio
    async def test_search_no_results(self, test_config: AppConfig, mock_retriever_with_results):
        """Test search with no matching documents."""
        mock_retriever = mock_retriever_with_results([], "")
        mock_ollama = Mock()  # Shouldn't be called

        agent = SearchAgent(
            config=test_config,
            ollama_client=mock_ollama,
            retriever=mock_retriever
        )

        result = await agent.search("something not in knowledge base")

        assert "couldn't find" in result.answer.lower() or "no" in result.answer.lower()
        assert result.source_document == "none"
        assert len(result.relevant_passages) == 0

    @pytest.mark.asyncio
    async def test_search_multiple_passages(self, test_config: AppConfig, mock_retriever_with_results, mock_ollama_with_response):
        """Test search with multiple relevant passages."""
        passages = [
            RetrievalResult(
                document_name="fees_charges.md",
                content="Overdraft fee is $35.",
                score=0.1,
                chunk_id="fees_charges.md_0"
            ),
            RetrievalResult(
                document_name="fees_charges.md",
                content="Refund policy allows one courtesy refund.",
                score=0.2,
                chunk_id="fees_charges.md_1"
            )
        ]
        context = "[Source 1: fees_charges.md]\nOverdraft fee is $35.\n\n---\n\n[Source 2: fees_charges.md]\nRefund policy allows one courtesy refund."

        mock_retriever = mock_retriever_with_results(passages, context)
        mock_ollama = mock_ollama_with_response({
            "answer": "The overdraft fee is $35. However, the refund policy allows one courtesy refund.",
            "primary_source": "fees_charges.md"
        })

        agent = SearchAgent(
            config=test_config,
            ollama_client=mock_ollama,
            retriever=mock_retriever
        )

        result = await agent.search("overdraft fees and refunds")

        assert len(result.relevant_passages) == 2
        assert result.source_document == "fees_charges.md"

    @pytest.mark.asyncio
    async def test_search_handles_invalid_json(self, test_config: AppConfig, mock_retriever_with_results):
        """Test handling of invalid JSON response from LLM."""
        passages = [
            RetrievalResult(
                document_name="test.md",
                content="Test content",
                score=0.1,
                chunk_id="test.md_0"
            )
        ]

        mock_retriever = mock_retriever_with_results(passages, "Test content")
        mock_ollama = Mock()
        mock_ollama.chat.return_value = {
            "message": {
                "content": "This is a plain text answer without JSON"
            }
        }

        agent = SearchAgent(
            config=test_config,
            ollama_client=mock_ollama,
            retriever=mock_retriever
        )

        result = await agent.search("test query")

        # Should use raw response as answer
        assert result.answer == "This is a plain text answer without JSON"
        assert result.source_document == "test.md"  # Falls back to first result

    @pytest.mark.asyncio
    async def test_search_handles_missing_source(self, test_config: AppConfig, mock_retriever_with_results, mock_ollama_with_response):
        """Test handling when LLM response has missing source."""
        passages = [
            RetrievalResult(
                document_name="real_doc.md",
                content="Real content",
                score=0.1,
                chunk_id="real_doc.md_0"
            )
        ]

        mock_retriever = mock_retriever_with_results(passages, "Real content")
        mock_ollama = mock_ollama_with_response({
            "answer": "Here is the answer",
            "primary_source": "none"  # LLM says no source
        })

        agent = SearchAgent(
            config=test_config,
            ollama_client=mock_ollama,
            retriever=mock_retriever
        )

        result = await agent.search("test query")

        # Should fall back to actual retrieved document
        assert result.source_document == "real_doc.md"

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, test_config: AppConfig, mock_ollama_with_response):
        """Test that search passes top_k to retriever."""
        mock_retriever = Mock(spec=DocumentRetriever)
        mock_retriever.search_with_context.return_value = ([], "")

        mock_ollama = mock_ollama_with_response({
            "answer": "No results",
            "primary_source": "none"
        })

        agent = SearchAgent(
            config=test_config,
            ollama_client=mock_ollama,
            retriever=mock_retriever
        )

        await agent.search("test query", top_k=5)

        mock_retriever.search_with_context.assert_called_once_with("test query", 5)

    def test_search_sync(self, test_config: AppConfig, mock_retriever_with_results, mock_ollama_with_response):
        """Test synchronous search method."""
        passages = [
            RetrievalResult(
                document_name="doc.md",
                content="Content",
                score=0.1,
                chunk_id="doc.md_0"
            )
        ]

        mock_retriever = mock_retriever_with_results(passages, "Content")
        mock_ollama = mock_ollama_with_response({
            "answer": "Sync answer",
            "primary_source": "doc.md"
        })

        agent = SearchAgent(
            config=test_config,
            ollama_client=mock_ollama,
            retriever=mock_retriever
        )

        result = agent.search_sync("test")

        assert result.answer == "Sync answer"

    @pytest.mark.asyncio
    async def test_search_builds_correct_prompt(self, test_config: AppConfig, mock_retriever_with_results):
        """Test that search builds the correct prompt for LLM."""
        passages = [
            RetrievalResult(
                document_name="test.md",
                content="Test content here",
                score=0.1,
                chunk_id="test.md_0"
            )
        ]
        context = "[Source 1: test.md]\nTest content here"

        mock_retriever = mock_retriever_with_results(passages, context)
        mock_ollama = Mock()
        mock_ollama.chat.return_value = {
            "message": {
                "content": '{"answer": "test", "primary_source": "test.md"}'
            }
        }

        agent = SearchAgent(
            config=test_config,
            ollama_client=mock_ollama,
            retriever=mock_retriever
        )

        await agent.search("my question")

        # Check the prompt contains context and question
        call_args = mock_ollama.chat.call_args
        user_message = call_args[1]["messages"][1]["content"]
        assert "Test content here" in user_message
        assert "my question" in user_message
