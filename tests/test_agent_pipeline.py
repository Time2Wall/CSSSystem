"""Integration tests for the Agent Pipeline."""

import pytest
import json
from unittest.mock import Mock, AsyncMock

from app.agents.pipeline import AgentPipeline, PipelineResult
from app.agents.reformulation import ReformulationAgent, ReformulationResult
from app.agents.search import SearchAgent, SearchResult
from app.agents.validation import ValidationAgent, ValidationResult
from app.rag.retriever import DocumentRetriever, RetrievalResult
from app.config import AppConfig


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_result_creation(self):
        """Test creating a pipeline result."""
        result = PipelineResult(
            original_question="test question",
            reformulated_query="reformulated",
            detected_intent="ACCOUNT",
            answer="test answer",
            source_document="test.md",
            confidence_score=85
        )
        assert result.original_question == "test question"
        assert result.confidence_score == 85

    def test_confidence_level_high(self):
        """Test high confidence level."""
        result = PipelineResult(
            original_question="q",
            reformulated_query="r",
            detected_intent="OTHER",
            answer="a",
            source_document="d",
            confidence_score=85
        )
        assert result.confidence_level == "high"

    def test_confidence_level_medium(self):
        """Test medium confidence level."""
        result = PipelineResult(
            original_question="q",
            reformulated_query="r",
            detected_intent="OTHER",
            answer="a",
            source_document="d",
            confidence_score=55
        )
        assert result.confidence_level == "medium"

    def test_confidence_level_low(self):
        """Test low confidence level."""
        result = PipelineResult(
            original_question="q",
            reformulated_query="r",
            detected_intent="OTHER",
            answer="a",
            source_document="d",
            confidence_score=25
        )
        assert result.confidence_level == "low"


class TestAgentPipeline:
    """Tests for AgentPipeline class."""

    @pytest.fixture
    def mock_reformulation_agent(self):
        """Create mock reformulation agent."""
        agent = Mock(spec=ReformulationAgent)

        async def mock_reformulate(question):
            return ReformulationResult(
                original_question=question,
                reformulated_query="reformulated: " + question,
                detected_intent="ACCOUNT"
            )

        agent.reformulate = AsyncMock(side_effect=mock_reformulate)
        return agent

    @pytest.fixture
    def mock_search_agent(self):
        """Create mock search agent."""
        agent = Mock(spec=SearchAgent)

        async def mock_search(query, top_k=None):
            return SearchResult(
                query=query,
                answer="This is the answer based on the knowledge base.",
                source_document="account_opening.md",
                relevant_passages=[
                    RetrievalResult(
                        document_name="account_opening.md",
                        content="Account opening requires valid ID.",
                        score=0.1,
                        chunk_id="account_opening.md_0"
                    )
                ]
            )

        agent.search = AsyncMock(side_effect=mock_search)
        return agent

    @pytest.fixture
    def mock_validation_agent(self):
        """Create mock validation agent."""
        agent = Mock(spec=ValidationAgent)

        async def mock_validate(question, answer, sources):
            return ValidationResult(
                confidence_score=85,
                reasoning="Good answer",
                is_grounded=True,
                is_relevant=True,
                is_complete=True
            )

        agent.validate = AsyncMock(side_effect=mock_validate)
        return agent

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_full_flow(
        self,
        test_config: AppConfig,
        mock_reformulation_agent,
        mock_search_agent,
        mock_validation_agent
    ):
        """Test complete pipeline flow."""
        pipeline = AgentPipeline(
            config=test_config,
            reformulation_agent=mock_reformulation_agent,
            search_agent=mock_search_agent,
            validation_agent=mock_validation_agent
        )

        result = await pipeline.process("How do I open an account?")

        # Verify all agents were called
        mock_reformulation_agent.reformulate.assert_called_once_with("How do I open an account?")
        mock_search_agent.search.assert_called_once()
        mock_validation_agent.validate.assert_called_once()

        # Verify result contains all expected fields
        assert result.original_question == "How do I open an account?"
        assert result.reformulated_query.startswith("reformulated:")
        assert result.detected_intent == "ACCOUNT"
        assert result.answer is not None
        assert result.source_document == "account_opening.md"
        assert result.confidence_score == 85
        assert result.confidence_level == "high"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_timing_recorded(
        self,
        test_config: AppConfig,
        mock_reformulation_agent,
        mock_search_agent,
        mock_validation_agent
    ):
        """Test that timing information is recorded."""
        pipeline = AgentPipeline(
            config=test_config,
            reformulation_agent=mock_reformulation_agent,
            search_agent=mock_search_agent,
            validation_agent=mock_validation_agent
        )

        result = await pipeline.process("test question")

        # Timing should be non-negative
        assert result.total_time_ms >= 0
        assert result.reformulation_time_ms >= 0
        assert result.search_time_ms >= 0
        assert result.validation_time_ms >= 0

        # Total should be at least sum of parts (approximately)
        sum_parts = result.reformulation_time_ms + result.search_time_ms + result.validation_time_ms
        assert result.total_time_ms >= sum_parts - 10  # Allow small tolerance

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_passes_reformulated_query_to_search(
        self,
        test_config: AppConfig,
        mock_reformulation_agent,
        mock_search_agent,
        mock_validation_agent
    ):
        """Test that reformulated query is passed to search agent."""
        pipeline = AgentPipeline(
            config=test_config,
            reformulation_agent=mock_reformulation_agent,
            search_agent=mock_search_agent,
            validation_agent=mock_validation_agent
        )

        await pipeline.process("original question")

        # Search should receive the reformulated query
        search_call = mock_search_agent.search.call_args
        assert "reformulated:" in search_call[0][0]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_passes_sources_to_validation(
        self,
        test_config: AppConfig,
        mock_reformulation_agent,
        mock_search_agent,
        mock_validation_agent
    ):
        """Test that source passages are passed to validation agent."""
        pipeline = AgentPipeline(
            config=test_config,
            reformulation_agent=mock_reformulation_agent,
            search_agent=mock_search_agent,
            validation_agent=mock_validation_agent
        )

        await pipeline.process("test question")

        # Validation should receive sources
        validation_call = mock_validation_agent.validate.call_args
        sources = validation_call[1]["sources"]
        assert len(sources) > 0
        assert "Account opening requires valid ID" in sources[0]

    def test_pipeline_sync(
        self,
        test_config: AppConfig,
        mock_reformulation_agent,
        mock_search_agent,
        mock_validation_agent
    ):
        """Test synchronous pipeline method."""
        pipeline = AgentPipeline(
            config=test_config,
            reformulation_agent=mock_reformulation_agent,
            search_agent=mock_search_agent,
            validation_agent=mock_validation_agent
        )

        result = pipeline.process_sync("sync test")

        assert result.original_question == "sync test"
        assert result.answer is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_preserves_passages(
        self,
        test_config: AppConfig,
        mock_reformulation_agent,
        mock_search_agent,
        mock_validation_agent
    ):
        """Test that relevant passages are preserved in result."""
        pipeline = AgentPipeline(
            config=test_config,
            reformulation_agent=mock_reformulation_agent,
            search_agent=mock_search_agent,
            validation_agent=mock_validation_agent
        )

        result = await pipeline.process("test")

        assert len(result.relevant_passages) == 1
        assert result.relevant_passages[0].document_name == "account_opening.md"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pipeline_with_custom_ollama_client(self, test_config: AppConfig, mock_chroma_client):
        """Test pipeline with custom Ollama client."""
        mock_ollama = Mock()

        # Configure responses for each agent
        call_count = [0]

        def mock_chat(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # Reformulation
                return {"message": {"content": '{"reformulated_query": "query", "detected_intent": "OTHER"}'}}
            elif call_count[0] == 2:  # Search
                return {"message": {"content": '{"answer": "answer", "primary_source": "doc.md"}'}}
            else:  # Validation
                return {"message": {"content": '{"grounded_score": 30, "relevant_score": 20, "complete_score": 15, "clear_score": 8, "is_grounded": true, "is_relevant": true, "is_complete": true, "reasoning": "ok"}'}}

        mock_ollama.chat.side_effect = mock_chat
        mock_ollama.embeddings.return_value = {"embedding": [0.1] * 768}

        # Index some test data
        from app.rag.indexer import DocumentIndexer, Chunk
        indexer = DocumentIndexer(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )
        indexer.index_chunks([
            Chunk(document_name="doc.md", content="Test content", chunk_index=0, metadata={"source": "doc.md"})
        ])

        pipeline = AgentPipeline(
            config=test_config,
            ollama_client=mock_ollama
        )
        # Need to set up the retriever with the same chroma client
        from app.rag.retriever import DocumentRetriever
        pipeline.retriever = DocumentRetriever(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )
        pipeline.search_agent.retriever = pipeline.retriever

        result = await pipeline.process("test question")

        assert result.answer == "answer"
        assert call_count[0] >= 3  # All three agents called
