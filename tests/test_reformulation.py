"""Unit tests for the Reformulation Agent."""

import pytest
import json
from unittest.mock import Mock, AsyncMock

from app.agents.reformulation import ReformulationAgent, ReformulationResult
from app.config import AppConfig


class TestReformulationResult:
    """Tests for ReformulationResult dataclass."""

    def test_result_creation(self):
        """Test creating a reformulation result."""
        result = ReformulationResult(
            original_question="Customer is angry about fees",
            reformulated_query="account fees and refund policy",
            detected_intent="FEES"
        )
        assert result.original_question == "Customer is angry about fees"
        assert result.reformulated_query == "account fees and refund policy"
        assert result.detected_intent == "FEES"


class TestReformulationAgent:
    """Tests for ReformulationAgent class."""

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
    async def test_reformulate_basic(self, test_config: AppConfig, mock_ollama_with_response):
        """Test basic reformulation."""
        mock_ollama = mock_ollama_with_response({
            "reformulated_query": "checking account opening requirements",
            "detected_intent": "ACCOUNT"
        })

        agent = ReformulationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.reformulate("How do I open a checking account?")

        assert result.original_question == "How do I open a checking account?"
        assert result.reformulated_query == "checking account opening requirements"
        assert result.detected_intent == "ACCOUNT"

    @pytest.mark.asyncio
    async def test_reformulate_removes_emotional_language(self, test_config: AppConfig, mock_ollama_with_response):
        """Test that reformulation handles emotional language."""
        mock_ollama = mock_ollama_with_response({
            "reformulated_query": "unauthorized credit card charge dispute process",
            "detected_intent": "CARDS"
        })

        agent = ReformulationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.reformulate("Customer is screaming that someone stole money from their card!")

        assert "screaming" not in result.reformulated_query.lower()
        assert result.detected_intent == "CARDS"

    @pytest.mark.asyncio
    async def test_reformulate_handles_various_intents(self, test_config: AppConfig, mock_ollama_with_response):
        """Test reformulation with different intent categories."""
        test_cases = [
            ("ACCOUNT", "account opening"),
            ("LOANS", "mortgage application"),
            ("FEES", "overdraft fee refund"),
            ("CARDS", "credit card fraud"),
            ("BRANCH", "branch hours"),
            ("TECH", "mobile app login"),
            ("OTHER", "general inquiry"),
        ]

        for intent, query in test_cases:
            mock_ollama = mock_ollama_with_response({
                "reformulated_query": query,
                "detected_intent": intent
            })

            agent = ReformulationAgent(config=test_config, ollama_client=mock_ollama)
            result = await agent.reformulate("test question")

            assert result.detected_intent == intent, f"Failed for intent {intent}"

    @pytest.mark.asyncio
    async def test_reformulate_handles_invalid_json(self, test_config: AppConfig):
        """Test handling of invalid JSON response from LLM."""
        mock_ollama = Mock()
        mock_ollama.chat.return_value = {
            "message": {
                "content": "This is not valid JSON"
            }
        }

        agent = ReformulationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.reformulate("test question")

        # Should fallback gracefully
        assert result.original_question == "test question"
        assert result.reformulated_query == "test question"  # Falls back to original
        assert result.detected_intent == "OTHER"

    @pytest.mark.asyncio
    async def test_reformulate_handles_invalid_intent(self, test_config: AppConfig, mock_ollama_with_response):
        """Test handling of invalid intent category."""
        mock_ollama = mock_ollama_with_response({
            "reformulated_query": "test query",
            "detected_intent": "INVALID_INTENT"
        })

        agent = ReformulationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.reformulate("test question")

        assert result.detected_intent == "OTHER"  # Invalid intent becomes OTHER

    @pytest.mark.asyncio
    async def test_reformulate_handles_partial_json(self, test_config: AppConfig):
        """Test handling of partial JSON response."""
        mock_ollama = Mock()
        mock_ollama.chat.return_value = {
            "message": {
                "content": '{"reformulated_query": "test query"}'  # Missing intent
            }
        }

        agent = ReformulationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.reformulate("test question")

        assert result.reformulated_query == "test query"
        assert result.detected_intent == "OTHER"  # Default for missing intent

    @pytest.mark.asyncio
    async def test_reformulate_extracts_json_from_text(self, test_config: AppConfig):
        """Test extracting JSON embedded in text response."""
        mock_ollama = Mock()
        mock_ollama.chat.return_value = {
            "message": {
                "content": 'Here is the result: {"reformulated_query": "extracted query", "detected_intent": "TECH"} Hope this helps!'
            }
        }

        agent = ReformulationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.reformulate("test question")

        assert result.reformulated_query == "extracted query"
        assert result.detected_intent == "TECH"

    @pytest.mark.asyncio
    async def test_reformulate_preserves_original_question(self, test_config: AppConfig, mock_ollama_with_response):
        """Test that original question is always preserved."""
        original = "This is my original question with special chars: @#$%"

        mock_ollama = mock_ollama_with_response({
            "reformulated_query": "cleaned query",
            "detected_intent": "OTHER"
        })

        agent = ReformulationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.reformulate(original)

        assert result.original_question == original

    def test_reformulate_sync(self, test_config: AppConfig, mock_ollama_with_response):
        """Test synchronous reformulation method."""
        mock_ollama = mock_ollama_with_response({
            "reformulated_query": "sync test query",
            "detected_intent": "ACCOUNT"
        })

        agent = ReformulationAgent(config=test_config, ollama_client=mock_ollama)
        result = agent.reformulate_sync("test question")

        assert result.reformulated_query == "sync test query"
        assert result.detected_intent == "ACCOUNT"

    @pytest.mark.asyncio
    async def test_reformulate_calls_llm_correctly(self, test_config: AppConfig):
        """Test that LLM is called with correct parameters."""
        mock_ollama = Mock()
        mock_ollama.chat.return_value = {
            "message": {
                "content": '{"reformulated_query": "test", "detected_intent": "OTHER"}'
            }
        }

        agent = ReformulationAgent(config=test_config, ollama_client=mock_ollama)
        await agent.reformulate("test question")

        # Verify LLM was called
        mock_ollama.chat.assert_called_once()

        # Check the call arguments
        call_args = mock_ollama.chat.call_args
        assert call_args[1]["model"] == test_config.ollama.llm_model
        assert len(call_args[1]["messages"]) == 2
        assert call_args[1]["messages"][0]["role"] == "system"
        assert call_args[1]["messages"][1]["role"] == "user"
        assert "test question" in call_args[1]["messages"][1]["content"]
