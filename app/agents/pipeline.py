"""Agent Pipeline - orchestrates all agents in sequence."""

import time
from dataclasses import dataclass, field
from typing import Optional
import ollama

from app.config import AppConfig, get_config
from app.agents.reformulation import ReformulationAgent, ReformulationResult
from app.agents.search import SearchAgent, SearchResult
from app.agents.validation import ValidationAgent, ValidationResult
from app.rag.retriever import DocumentRetriever, RetrievalResult


@dataclass
class PipelineResult:
    """Complete result from the agent pipeline."""
    # Input
    original_question: str

    # Reformulation stage
    reformulated_query: str
    detected_intent: str

    # Search stage
    answer: str
    source_document: str
    relevant_passages: list[RetrievalResult] = field(default_factory=list)

    # Validation stage
    confidence_score: int = 0
    validation_reasoning: str = ""
    is_grounded: bool = False
    is_relevant: bool = False
    is_complete: bool = False

    # Timing
    total_time_ms: int = 0
    reformulation_time_ms: int = 0
    search_time_ms: int = 0
    validation_time_ms: int = 0

    @property
    def confidence_level(self) -> str:
        """Get a human-readable confidence level."""
        if self.confidence_score >= 70:
            return "high"
        elif self.confidence_score >= 40:
            return "medium"
        else:
            return "low"


class AgentPipeline:
    """Orchestrates the multi-agent system for answering questions."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        ollama_client: Optional[ollama.Client] = None,
        retriever: Optional[DocumentRetriever] = None,
        reformulation_agent: Optional[ReformulationAgent] = None,
        search_agent: Optional[SearchAgent] = None,
        validation_agent: Optional[ValidationAgent] = None,
    ):
        """Initialize the agent pipeline.

        Args:
            config: Application configuration
            ollama_client: Shared Ollama client
            retriever: Document retriever
            reformulation_agent: Custom reformulation agent
            search_agent: Custom search agent
            validation_agent: Custom validation agent
        """
        self.config = config or get_config()
        self.ollama_client = ollama_client or ollama.Client(host=self.config.ollama.host)

        self.retriever = retriever or DocumentRetriever(
            config=self.config,
            ollama_client=self.ollama_client
        )

        self.reformulation_agent = reformulation_agent or ReformulationAgent(
            config=self.config,
            ollama_client=self.ollama_client
        )

        self.search_agent = search_agent or SearchAgent(
            config=self.config,
            ollama_client=self.ollama_client,
            retriever=self.retriever
        )

        self.validation_agent = validation_agent or ValidationAgent(
            config=self.config,
            ollama_client=self.ollama_client
        )

    async def process(self, question: str) -> PipelineResult:
        """Process a question through all agents.

        Args:
            question: The raw question from the representative

        Returns:
            PipelineResult with complete answer and metadata
        """
        total_start = time.time()

        # Stage 1: Reformulation
        reform_start = time.time()
        reformulation_result: ReformulationResult = await self.reformulation_agent.reformulate(question)
        reformulation_time_ms = int((time.time() - reform_start) * 1000)

        # Stage 2: Search
        search_start = time.time()
        search_result: SearchResult = await self.search_agent.search(
            reformulation_result.reformulated_query
        )
        search_time_ms = int((time.time() - search_start) * 1000)

        # Stage 3: Validation
        validation_start = time.time()
        sources = [passage.content for passage in search_result.relevant_passages]
        validation_result: ValidationResult = await self.validation_agent.validate(
            question=reformulation_result.original_question,
            answer=search_result.answer,
            sources=sources
        )
        validation_time_ms = int((time.time() - validation_start) * 1000)

        total_time_ms = int((time.time() - total_start) * 1000)

        return PipelineResult(
            # Input
            original_question=question,

            # Reformulation
            reformulated_query=reformulation_result.reformulated_query,
            detected_intent=reformulation_result.detected_intent,

            # Search
            answer=search_result.answer,
            source_document=search_result.source_document,
            relevant_passages=search_result.relevant_passages,

            # Validation
            confidence_score=validation_result.confidence_score,
            validation_reasoning=validation_result.reasoning,
            is_grounded=validation_result.is_grounded,
            is_relevant=validation_result.is_relevant,
            is_complete=validation_result.is_complete,

            # Timing
            total_time_ms=total_time_ms,
            reformulation_time_ms=reformulation_time_ms,
            search_time_ms=search_time_ms,
            validation_time_ms=validation_time_ms,
        )

    def process_sync(self, question: str) -> PipelineResult:
        """Synchronous version of process.

        Args:
            question: The raw question

        Returns:
            PipelineResult with complete answer and metadata
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.process(question))
