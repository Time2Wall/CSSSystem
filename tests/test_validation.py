"""Unit tests for the Validation Agent."""

import pytest
import json
from unittest.mock import Mock

from app.agents.validation import ValidationAgent, ValidationResult
from app.config import AppConfig


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(
            confidence_score=85,
            reasoning="Well grounded answer",
            is_grounded=True,
            is_relevant=True,
            is_complete=True
        )
        assert result.confidence_score == 85
        assert result.reasoning == "Well grounded answer"
        assert result.is_grounded is True
        assert result.is_relevant is True
        assert result.is_complete is True

    def test_result_with_low_confidence(self):
        """Test result with low confidence."""
        result = ValidationResult(
            confidence_score=25,
            reasoning="Poorly supported",
            is_grounded=False,
            is_relevant=True,
            is_complete=False
        )
        assert result.confidence_score == 25
        assert result.is_grounded is False


class TestValidationAgent:
    """Tests for ValidationAgent class."""

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
    async def test_validate_high_quality(self, test_config: AppConfig, mock_ollama_with_response):
        """Test validation of high-quality answer."""
        mock_ollama = mock_ollama_with_response({
            "grounded_score": 38,
            "relevant_score": 28,
            "complete_score": 18,
            "clear_score": 9,
            "is_grounded": True,
            "is_relevant": True,
            "is_complete": True,
            "reasoning": "Excellent answer fully supported by sources"
        })

        agent = ValidationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.validate(
            question="How do I open an account?",
            answer="To open an account, you need valid ID and minimum deposit of $25.",
            sources=["To open a checking account: Valid ID, Minimum deposit $25"]
        )

        assert result.confidence_score == 93
        assert result.is_grounded is True
        assert result.is_relevant is True
        assert result.is_complete is True

    @pytest.mark.asyncio
    async def test_validate_low_quality(self, test_config: AppConfig, mock_ollama_with_response):
        """Test validation of low-quality answer."""
        mock_ollama = mock_ollama_with_response({
            "grounded_score": 10,
            "relevant_score": 10,
            "complete_score": 5,
            "clear_score": 3,
            "is_grounded": False,
            "is_relevant": False,
            "is_complete": False,
            "reasoning": "Answer not well supported by sources"
        })

        agent = ValidationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.validate(
            question="What is the interest rate?",
            answer="The interest rate is probably around 5%.",
            sources=["We offer various loan products."]
        )

        assert result.confidence_score == 28
        assert result.is_grounded is False
        assert result.is_relevant is False

    @pytest.mark.asyncio
    async def test_validate_handles_invalid_json(self, test_config: AppConfig):
        """Test handling of invalid JSON response."""
        mock_ollama = Mock()
        mock_ollama.chat.return_value = {
            "message": {
                "content": "This is not valid JSON"
            }
        }

        agent = ValidationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.validate(
            question="test",
            answer="test answer",
            sources=["source"]
        )

        # Should fall back to moderate confidence
        assert result.confidence_score == 50
        assert "Unable to parse" in result.reasoning

    @pytest.mark.asyncio
    async def test_validate_caps_scores(self, test_config: AppConfig, mock_ollama_with_response):
        """Test that scores are capped at maximum values."""
        mock_ollama = mock_ollama_with_response({
            "grounded_score": 100,  # Over max of 40
            "relevant_score": 50,   # Over max of 30
            "complete_score": 30,   # Over max of 20
            "clear_score": 20,      # Over max of 10
            "is_grounded": True,
            "is_relevant": True,
            "is_complete": True,
            "reasoning": "Perfect score"
        })

        agent = ValidationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.validate(
            question="test",
            answer="answer",
            sources=["source"]
        )

        # Should be capped at max: 40 + 30 + 20 + 10 = 100
        assert result.confidence_score == 100

    @pytest.mark.asyncio
    async def test_validate_floors_scores(self, test_config: AppConfig, mock_ollama_with_response):
        """Test that scores are floored at minimum values."""
        mock_ollama = mock_ollama_with_response({
            "grounded_score": -10,  # Below min of 0
            "relevant_score": -5,
            "complete_score": -3,
            "clear_score": -1,
            "is_grounded": False,
            "is_relevant": False,
            "is_complete": False,
            "reasoning": "Terrible score"
        })

        agent = ValidationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.validate(
            question="test",
            answer="answer",
            sources=["source"]
        )

        # Should be floored at 0
        assert result.confidence_score == 0

    @pytest.mark.asyncio
    async def test_validate_empty_sources(self, test_config: AppConfig, mock_ollama_with_response):
        """Test validation with empty sources list."""
        mock_ollama = mock_ollama_with_response({
            "grounded_score": 10,
            "relevant_score": 15,
            "complete_score": 10,
            "clear_score": 5,
            "is_grounded": False,
            "is_relevant": True,
            "is_complete": False,
            "reasoning": "No sources to verify against"
        })

        agent = ValidationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.validate(
            question="test",
            answer="answer",
            sources=[]
        )

        # Should still work, likely lower confidence
        assert result.confidence_score == 40
        assert result.is_grounded is False

    @pytest.mark.asyncio
    async def test_validate_partial_json_response(self, test_config: AppConfig):
        """Test handling of partial JSON response."""
        mock_ollama = Mock()
        mock_ollama.chat.return_value = {
            "message": {
                "content": '{"grounded_score": 30, "relevant_score": 25}'  # Missing fields
            }
        }

        agent = ValidationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.validate(
            question="test",
            answer="answer",
            sources=["source"]
        )

        # Should use defaults for missing fields
        assert result.confidence_score == 30 + 25 + 10 + 5  # 10 and 5 are defaults

    def test_validate_sync(self, test_config: AppConfig, mock_ollama_with_response):
        """Test synchronous validation method."""
        mock_ollama = mock_ollama_with_response({
            "grounded_score": 35,
            "relevant_score": 25,
            "complete_score": 15,
            "clear_score": 8,
            "is_grounded": True,
            "is_relevant": True,
            "is_complete": True,
            "reasoning": "Good answer"
        })

        agent = ValidationAgent(config=test_config, ollama_client=mock_ollama)
        result = agent.validate_sync(
            question="test",
            answer="answer",
            sources=["source"]
        )

        assert result.confidence_score == 83

    @pytest.mark.asyncio
    async def test_validate_builds_correct_prompt(self, test_config: AppConfig):
        """Test that validation builds the correct prompt."""
        mock_ollama = Mock()
        mock_ollama.chat.return_value = {
            "message": {
                "content": '{"grounded_score": 30, "relevant_score": 20, "complete_score": 15, "clear_score": 7, "is_grounded": true, "is_relevant": true, "is_complete": true, "reasoning": "ok"}'
            }
        }

        agent = ValidationAgent(config=test_config, ollama_client=mock_ollama)
        await agent.validate(
            question="My question",
            answer="My answer",
            sources=["Source 1", "Source 2"]
        )

        # Check the prompt contains all elements
        call_args = mock_ollama.chat.call_args
        user_message = call_args[1]["messages"][1]["content"]
        assert "My question" in user_message
        assert "My answer" in user_message
        assert "Source 1" in user_message
        assert "Source 2" in user_message

    @pytest.mark.asyncio
    async def test_validate_extracts_json_from_text(self, test_config: AppConfig):
        """Test extracting JSON embedded in text response."""
        mock_ollama = Mock()
        mock_ollama.chat.return_value = {
            "message": {
                "content": 'Based on my analysis: {"grounded_score": 35, "relevant_score": 25, "complete_score": 18, "clear_score": 9, "is_grounded": true, "is_relevant": true, "is_complete": true, "reasoning": "Good"} End of evaluation.'
            }
        }

        agent = ValidationAgent(config=test_config, ollama_client=mock_ollama)
        result = await agent.validate(
            question="test",
            answer="answer",
            sources=["source"]
        )

        assert result.confidence_score == 87
        assert result.is_grounded is True
