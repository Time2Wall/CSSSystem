"""Search Agent - performs RAG search and generates grounded answers."""

import json
import re
from dataclasses import dataclass, field
from typing import Optional
import ollama

from app.config import AppConfig, get_config
from app.rag.retriever import DocumentRetriever, RetrievalResult


@dataclass
class SearchResult:
    """Result from the search agent."""
    query: str
    answer: str
    source_document: str
    relevant_passages: list[RetrievalResult] = field(default_factory=list)


SEARCH_SYSTEM_PROMPT = """You are a helpful bank customer service assistant. Your role is to answer questions
based ONLY on the provided context from the bank's knowledge base.

Guidelines:
1. Answer questions accurately using ONLY the information in the provided context
2. If the context doesn't contain enough information to answer, say so clearly
3. Be concise but complete - include all relevant details
4. Use a professional, helpful tone
5. If there are multiple relevant pieces of information, synthesize them into a coherent answer
6. Always cite which document(s) your answer is based on

You MUST respond in this exact JSON format:
{
    "answer": "your complete answer to the question",
    "primary_source": "the main document name your answer is based on"
}

If you cannot find relevant information in the context, respond with:
{
    "answer": "I don't have enough information in the knowledge base to answer this question. Please consult with a supervisor or refer to the official policy documents.",
    "primary_source": "none"
}
"""


class SearchAgent:
    """Agent that searches the knowledge base and generates grounded answers."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        ollama_client: Optional[ollama.Client] = None,
        retriever: Optional[DocumentRetriever] = None,
    ):
        """Initialize the search agent.

        Args:
            config: Application configuration
            ollama_client: Ollama client for LLM calls (optional, creates default)
            retriever: Document retriever for RAG search (optional, creates default)
        """
        self.config = config or get_config()
        self.ollama_client = ollama_client or ollama.Client(host=self.config.ollama.host)
        self.retriever = retriever or DocumentRetriever(config=self.config, ollama_client=self.ollama_client)

    async def search(self, query: str, top_k: Optional[int] = None) -> SearchResult:
        """Search the knowledge base and generate an answer.

        Args:
            query: The search query (ideally reformulated)
            top_k: Number of passages to retrieve

        Returns:
            SearchResult with answer, source document, and relevant passages
        """
        # Retrieve relevant passages
        results, context = self.retriever.search_with_context(query, top_k)

        if not results:
            return SearchResult(
                query=query,
                answer="I couldn't find any relevant information in the knowledge base for this query.",
                source_document="none",
                relevant_passages=[]
            )

        # Build the prompt with context
        user_prompt = f"""Context from knowledge base:

{context}

Question: {query}

Please provide a helpful answer based on the context above."""

        # Call the LLM
        response = self.ollama_client.chat(
            model=self.config.ollama.llm_model,
            messages=[
                {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.3}
        )

        response_text = response["message"]["content"]

        # Parse the JSON response
        try:
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                answer = parsed.get("answer", response_text)
                source_document = parsed.get("primary_source", results[0].document_name if results else "unknown")
            else:
                # Fallback: use the raw response as answer
                answer = response_text
                source_document = results[0].document_name if results else "unknown"
        except json.JSONDecodeError:
            answer = response_text
            source_document = results[0].document_name if results else "unknown"

        # Ensure source document is valid
        if source_document == "none" or not source_document:
            source_document = results[0].document_name if results else "unknown"

        return SearchResult(
            query=query,
            answer=answer,
            source_document=source_document,
            relevant_passages=results
        )

    def search_sync(self, query: str, top_k: Optional[int] = None) -> SearchResult:
        """Synchronous version of search.

        Args:
            query: The search query
            top_k: Number of passages to retrieve

        Returns:
            SearchResult with answer and sources
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.search(query, top_k))
