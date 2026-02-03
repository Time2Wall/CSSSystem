"""Reformulation Agent - rewrites customer questions into optimized search queries."""

import json
import re
from dataclasses import dataclass
from typing import Optional
import ollama

from app.config import AppConfig, get_config


@dataclass
class ReformulationResult:
    """Result from the reformulation agent."""
    original_question: str
    reformulated_query: str
    detected_intent: str


REFORMULATION_SYSTEM_PROMPT = """You are a query reformulation specialist for a bank customer service system.

Your task is to take raw questions from customer service representatives (which may include emotional language,
incomplete sentences, or informal descriptions) and convert them into clear, optimized search queries.

Guidelines:
1. Remove emotional language (e.g., "customer is angry", "they're yelling")
2. Extract the core banking topic or issue
3. Convert to a clear, search-friendly query
4. Identify the main intent category

Intent categories:
- ACCOUNT: Account opening, closing, management
- LOANS: Personal loans, mortgages, auto loans
- FEES: Fees, charges, refunds, disputes
- CARDS: Credit cards, debit cards, fraud
- BRANCH: Branch locations, hours, contact info
- TECH: Mobile app, online banking, technical issues
- OTHER: Anything else

You MUST respond in this exact JSON format:
{
    "reformulated_query": "the optimized search query",
    "detected_intent": "one of the intent categories above"
}

Examples:

Input: "Customer is yelling that money was stolen from his card"
Output: {"reformulated_query": "unauthorized credit card charge dispute process and fraud protection", "detected_intent": "CARDS"}

Input: "how do I help someone open a checking account they're in a rush"
Output: {"reformulated_query": "checking account opening requirements and process", "detected_intent": "ACCOUNT"}

Input: "app won't let them log in keeps saying error"
Output: {"reformulated_query": "mobile app login error troubleshooting", "detected_intent": "TECH"}
"""


class ReformulationAgent:
    """Agent that reformulates customer questions into optimized search queries."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        ollama_client: Optional[ollama.Client] = None,
    ):
        """Initialize the reformulation agent.

        Args:
            config: Application configuration
            ollama_client: Ollama client for LLM calls (optional, creates default)
        """
        self.config = config or get_config()
        self.ollama_client = ollama_client or ollama.Client(host=self.config.ollama.host)

    async def reformulate(self, raw_question: str) -> ReformulationResult:
        """Reformulate a raw question into an optimized search query.

        Args:
            raw_question: The raw question from the representative

        Returns:
            ReformulationResult with original, reformulated query, and intent
        """
        # Call the LLM
        response = self.ollama_client.chat(
            model=self.config.ollama.llm_model,
            messages=[
                {"role": "system", "content": REFORMULATION_SYSTEM_PROMPT},
                {"role": "user", "content": raw_question}
            ],
            options={"temperature": 0.3}  # Lower temperature for more consistent output
        )

        response_text = response["message"]["content"]

        # Parse the JSON response
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                reformulated_query = parsed.get("reformulated_query", raw_question)
                detected_intent = parsed.get("detected_intent", "OTHER")
            else:
                # Fallback if no JSON found
                reformulated_query = raw_question
                detected_intent = "OTHER"
        except json.JSONDecodeError:
            # Fallback on parse error
            reformulated_query = raw_question
            detected_intent = "OTHER"

        # Validate intent category
        valid_intents = ["ACCOUNT", "LOANS", "FEES", "CARDS", "BRANCH", "TECH", "OTHER"]
        if detected_intent not in valid_intents:
            detected_intent = "OTHER"

        return ReformulationResult(
            original_question=raw_question,
            reformulated_query=reformulated_query,
            detected_intent=detected_intent
        )

    def reformulate_sync(self, raw_question: str) -> ReformulationResult:
        """Synchronous version of reformulate for simpler use cases.

        Args:
            raw_question: The raw question from the representative

        Returns:
            ReformulationResult with original, reformulated query, and intent
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.reformulate(raw_question))
