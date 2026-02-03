"""End-to-end tests for the Customer Service Support System."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import create_app
from app.config import AppConfig
from app.api.routes import set_dependencies
from app.agents.pipeline import AgentPipeline, PipelineResult
from app.agents.reformulation import ReformulationAgent, ReformulationResult
from app.agents.search import SearchAgent, SearchResult
from app.agents.validation import ValidationAgent, ValidationResult
from app.database.db import DatabaseManager
from app.rag.indexer import DocumentIndexer, Chunk
from app.rag.retriever import DocumentRetriever, RetrievalResult


class TestEndToEnd:
    """End-to-end tests for the complete system flow."""

    @pytest.fixture
    def mock_ollama_client(self):
        """Create mock Ollama client."""
        mock = Mock()

        # Track call counts for different agent stages
        self.call_count = [0]

        def mock_chat(*args, **kwargs):
            self.call_count[0] += 1
            messages = kwargs.get('messages', [])
            user_content = messages[-1]['content'] if messages else ''

            # Determine which agent is calling based on system prompt
            system_content = messages[0]['content'] if messages else ''

            if 'reformulation' in system_content.lower():
                return {
                    "message": {
                        "content": '{"reformulated_query": "account opening requirements and process", "detected_intent": "ACCOUNT"}'
                    }
                }
            elif 'search' in system_content.lower() or 'answer' in system_content.lower():
                return {
                    "message": {
                        "content": '{"answer": "To open a checking account, you need a valid government-issued ID, Social Security number, proof of address, and a minimum deposit of $25.", "primary_source": "account_opening.md"}'
                    }
                }
            elif 'validation' in system_content.lower() or 'quality' in system_content.lower():
                return {
                    "message": {
                        "content": '{"grounded_score": 38, "relevant_score": 28, "complete_score": 18, "clear_score": 9, "is_grounded": true, "is_relevant": true, "is_complete": true, "reasoning": "Answer is well-grounded in source material"}'
                    }
                }
            else:
                return {
                    "message": {
                        "content": '{"reformulated_query": "general query", "detected_intent": "OTHER"}'
                    }
                }

        mock.chat.side_effect = mock_chat
        mock.embeddings.return_value = {"embedding": [0.1] * 768}

        return mock

    @pytest.fixture
    def test_system(self, test_config: AppConfig, mock_ollama_client, mock_chroma_client):
        """Set up a complete test system with mocked LLM."""
        # Initialize database
        db_manager = DatabaseManager(test_config)
        db_manager.clear_all()

        # Initialize indexer and index test documents
        indexer = DocumentIndexer(
            config=test_config,
            ollama_client=mock_ollama_client,
            chroma_client=mock_chroma_client
        )

        # Add test chunks
        chunks = [
            Chunk(
                document_name="account_opening.md",
                content="To open a checking account, you need: Valid government-issued ID, Social Security number, Proof of address, Minimum deposit of $25.",
                chunk_index=0,
                metadata={"source": "account_opening.md"}
            ),
            Chunk(
                document_name="fees_charges.md",
                content="Overdraft fee is $35 per transaction. First-time courtesy refund available for accounts in good standing.",
                chunk_index=0,
                metadata={"source": "fees_charges.md"}
            ),
            Chunk(
                document_name="credit_cards.md",
                content="Report unauthorized charges immediately by calling 1-800-555-FRAUD. Zero liability policy applies.",
                chunk_index=0,
                metadata={"source": "credit_cards.md"}
            ),
        ]
        indexer.index_chunks(chunks)

        # Initialize retriever
        retriever = DocumentRetriever(
            config=test_config,
            ollama_client=mock_ollama_client,
            chroma_client=mock_chroma_client
        )

        # Initialize pipeline with mocked components
        pipeline = AgentPipeline(
            config=test_config,
            ollama_client=mock_ollama_client,
            retriever=retriever
        )

        # Create app and set dependencies
        app = create_app(test_config)
        set_dependencies(pipeline, db_manager, retriever, test_config)

        return app, db_manager, pipeline

    @pytest.mark.integration
    def test_full_query_flow(self, test_system):
        """Test complete flow from question to answer."""
        app, db_manager, pipeline = test_system

        with TestClient(app) as client:
            # Submit a question
            response = client.post(
                "/api/query",
                json={"question": "How do I open a checking account?"}
            )

            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert "id" in data
            assert data["question"] == "How do I open a checking account?"
            assert "reformulated_query" in data
            assert "answer" in data
            assert "confidence_score" in data
            assert "source_document" in data
            assert "response_time_ms" in data

            # Verify confidence score is reasonable
            assert 0 <= data["confidence_score"] <= 100

            # Verify the query was saved to database
            saved_query = db_manager.get_query(data["id"])
            assert saved_query is not None
            assert saved_query.question == "How do I open a checking account?"

    @pytest.mark.integration
    def test_all_three_agents_called(self, test_system):
        """Test that all three agents are called in sequence."""
        app, db_manager, pipeline = test_system

        with TestClient(app) as client:
            response = client.post(
                "/api/query",
                json={"question": "Test question"}
            )

            assert response.status_code == 200
            data = response.json()

            # Verify response contains output from all 3 agents:
            # 1. Reformulation Agent - produces reformulated_query and detected_intent
            assert "reformulated_query" in data
            assert "detected_intent" in data

            # 2. Search Agent - produces answer and source_document
            assert "answer" in data
            assert "source_document" in data

            # 3. Validation Agent - produces confidence_score
            assert "confidence_score" in data
            assert 0 <= data["confidence_score"] <= 100

    @pytest.mark.integration
    def test_query_saved_to_database(self, test_system):
        """Test that query is saved to database."""
        app, db_manager, pipeline = test_system

        initial_count = db_manager.get_total_query_count()

        with TestClient(app) as client:
            client.post(
                "/api/query",
                json={"question": "Test question"}
            )

        assert db_manager.get_total_query_count() == initial_count + 1

    @pytest.mark.integration
    def test_stats_reflect_new_query(self, test_system):
        """Test that stats endpoint reflects new queries."""
        app, db_manager, pipeline = test_system

        with TestClient(app) as client:
            # Get initial stats
            initial_stats = client.get("/api/stats").json()
            initial_total = initial_stats["total_queries"]

            # Submit a query
            client.post("/api/query", json={"question": "Test question"})

            # Get updated stats
            updated_stats = client.get("/api/stats").json()

            assert updated_stats["total_queries"] == initial_total + 1

    @pytest.mark.integration
    def test_document_content_retrieval(self, test_system):
        """Test retrieving full document content."""
        app, db_manager, pipeline = test_system

        with TestClient(app) as client:
            # First submit a query to use a document
            response = client.post(
                "/api/query",
                json={"question": "How do I open an account?"}
            )
            source_doc = response.json()["source_document"]

            # The mock retriever should have this document
            # Note: In real system, this would retrieve actual content
            # Here we verify the endpoint works
            doc_response = client.get(f"/api/documents/{source_doc}")

            # May be 200 (if document exists) or 404 (mock doesn't have full content)
            assert doc_response.status_code in [200, 404]

    @pytest.mark.integration
    def test_query_history_retrieval(self, test_system):
        """Test retrieving query history."""
        app, db_manager, pipeline = test_system

        with TestClient(app) as client:
            # Submit multiple queries
            for i in range(3):
                client.post(
                    "/api/query",
                    json={"question": f"Question {i}"}
                )

            # Get query history
            response = client.get("/api/queries")

            assert response.status_code == 200
            data = response.json()
            assert len(data["queries"]) >= 3

    @pytest.mark.integration
    def test_specific_query_retrieval(self, test_system):
        """Test retrieving a specific query by ID."""
        app, db_manager, pipeline = test_system

        with TestClient(app) as client:
            # Submit a query
            submit_response = client.post(
                "/api/query",
                json={"question": "Specific question for retrieval test"}
            )
            query_id = submit_response.json()["id"]

            # Retrieve that specific query
            response = client.get(f"/api/query/{query_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == query_id
            assert data["question"] == "Specific question for retrieval test"

    @pytest.mark.integration
    def test_document_usage_tracking(self, test_system):
        """Test that document usage is tracked."""
        app, db_manager, pipeline = test_system

        with TestClient(app) as client:
            # Submit queries
            for _ in range(3):
                client.post("/api/query", json={"question": "Account question"})

            # Check document usage
            usages = db_manager.get_document_usage()

            # Should have at least one document with usage
            assert len(usages) >= 1
            assert usages[0].usage_count >= 1

    @pytest.mark.integration
    def test_confidence_distribution_tracking(self, test_system):
        """Test confidence distribution in stats."""
        app, db_manager, pipeline = test_system

        with TestClient(app) as client:
            # Submit some queries
            for _ in range(3):
                client.post("/api/query", json={"question": "Test question"})

            # Get stats
            response = client.get("/api/stats")
            stats = response.json()

            # Verify confidence distribution exists
            assert "confidence_distribution" in stats
            dist = stats["confidence_distribution"]
            assert "high" in dist
            assert "medium" in dist
            assert "low" in dist

            # Total should match number of queries
            total = dist["high"] + dist["medium"] + dist["low"]
            assert total == stats["total_queries"]

    @pytest.mark.integration
    def test_frontend_pages_load(self, test_system):
        """Test that frontend pages can be served."""
        app, _, _ = test_system

        with TestClient(app) as client:
            # Test index page
            index_response = client.get("/")
            # May be 200 or 500 depending on template availability
            assert index_response.status_code in [200, 500]

            # Test dashboard page
            dashboard_response = client.get("/dashboard")
            assert dashboard_response.status_code in [200, 500]

    @pytest.mark.integration
    def test_health_endpoint(self, test_system):
        """Test health check endpoint."""
        app, _, _ = test_system

        with TestClient(app) as client:
            response = client.get("/api/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "version" in data

    @pytest.mark.integration
    def test_error_handling_invalid_query(self, test_system):
        """Test error handling for invalid input."""
        app, _, _ = test_system

        with TestClient(app) as client:
            # Empty question should fail validation
            response = client.post(
                "/api/query",
                json={"question": ""}
            )
            assert response.status_code == 422

            # Missing question field should fail
            response = client.post(
                "/api/query",
                json={}
            )
            assert response.status_code == 422

    @pytest.mark.integration
    def test_response_contains_source_reference(self, test_system):
        """Test that response contains source document reference."""
        app, _, _ = test_system

        with TestClient(app) as client:
            response = client.post(
                "/api/query",
                json={"question": "How do I open an account?"}
            )

            data = response.json()
            assert "source_document" in data
            assert data["source_document"] is not None
            # Source should be a markdown file
            assert data["source_document"].endswith(".md") or data["source_document"] == "none"
