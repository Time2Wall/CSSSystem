"""Unit tests for the database module."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from app.database.models import Query, DocumentUsage
from app.database.db import DatabaseManager
from app.config import AppConfig


class TestQueryModel:
    """Tests for Query model."""

    def test_query_to_dict(self, test_config: AppConfig, tmp_path: Path):
        """Test Query to_dict method."""
        db = DatabaseManager(test_config)

        query = db.save_query(
            question="Test question",
            reformulated_query="Reformulated",
            detected_intent="ACCOUNT",
            answer="Test answer",
            confidence_score=85,
            source_document="test.md",
            response_time_ms=100
        )

        d = query.to_dict()

        assert d["id"] == query.id
        assert d["question"] == "Test question"
        assert d["reformulated_query"] == "Reformulated"
        assert d["detected_intent"] == "ACCOUNT"
        assert d["answer"] == "Test answer"
        assert d["confidence_score"] == 85
        assert d["source_document"] == "test.md"
        assert d["response_time_ms"] == 100
        assert d["created_at"] is not None


class TestDocumentUsageModel:
    """Tests for DocumentUsage model."""

    def test_document_usage_to_dict(self, test_config: AppConfig):
        """Test DocumentUsage to_dict method."""
        db = DatabaseManager(test_config)

        # Create usage by saving a query
        db.save_query(
            question="Test",
            reformulated_query="Test",
            detected_intent="OTHER",
            answer="Answer",
            confidence_score=50,
            source_document="test_doc.md"
        )

        usages = db.get_document_usage()
        assert len(usages) >= 1

        usage = usages[0]
        d = usage.to_dict()

        assert d["document_name"] == "test_doc.md"
        assert d["usage_count"] >= 1
        assert d["last_used"] is not None


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    def test_save_query(self, test_config: AppConfig):
        """Test saving a query."""
        db = DatabaseManager(test_config)

        query = db.save_query(
            question="How do I open an account?",
            reformulated_query="account opening requirements",
            detected_intent="ACCOUNT",
            answer="To open an account...",
            confidence_score=85,
            source_document="account_opening.md",
            response_time_ms=500
        )

        assert query.id is not None
        assert query.question == "How do I open an account?"
        assert query.confidence_score == 85

    def test_get_query(self, test_config: AppConfig):
        """Test getting a query by ID."""
        db = DatabaseManager(test_config)

        saved = db.save_query(
            question="Test",
            reformulated_query="Test",
            detected_intent="OTHER",
            answer="Answer",
            confidence_score=50,
            source_document="test.md"
        )

        retrieved = db.get_query(saved.id)

        assert retrieved is not None
        assert retrieved.id == saved.id
        assert retrieved.question == "Test"

    def test_get_query_not_found(self, test_config: AppConfig):
        """Test getting nonexistent query."""
        db = DatabaseManager(test_config)

        result = db.get_query(99999)

        assert result is None

    def test_get_queries_basic(self, test_config: AppConfig):
        """Test getting multiple queries."""
        db = DatabaseManager(test_config)
        db.clear_all()

        # Save multiple queries
        for i in range(5):
            db.save_query(
                question=f"Question {i}",
                reformulated_query=f"Query {i}",
                detected_intent="OTHER",
                answer=f"Answer {i}",
                confidence_score=50 + i * 10,
                source_document="test.md"
            )

        queries = db.get_queries(limit=10)

        assert len(queries) == 5

    def test_get_queries_with_limit(self, test_config: AppConfig):
        """Test query limit."""
        db = DatabaseManager(test_config)
        db.clear_all()

        for i in range(5):
            db.save_query(
                question=f"Q{i}",
                reformulated_query=f"R{i}",
                detected_intent="OTHER",
                answer=f"A{i}",
                confidence_score=50,
                source_document="test.md"
            )

        queries = db.get_queries(limit=3)

        assert len(queries) == 3

    def test_get_queries_with_offset(self, test_config: AppConfig):
        """Test query offset."""
        db = DatabaseManager(test_config)
        db.clear_all()

        for i in range(5):
            db.save_query(
                question=f"Q{i}",
                reformulated_query=f"R{i}",
                detected_intent="OTHER",
                answer=f"A{i}",
                confidence_score=50,
                source_document="test.md"
            )

        queries = db.get_queries(limit=10, offset=2)

        assert len(queries) == 3

    def test_get_queries_filter_by_intent(self, test_config: AppConfig):
        """Test filtering by intent."""
        db = DatabaseManager(test_config)
        db.clear_all()

        db.save_query("Q1", "R1", "ACCOUNT", "A1", 50, "test.md")
        db.save_query("Q2", "R2", "LOANS", "A2", 50, "test.md")
        db.save_query("Q3", "R3", "ACCOUNT", "A3", 50, "test.md")

        account_queries = db.get_queries(intent_filter="ACCOUNT")

        assert len(account_queries) == 2
        assert all(q.detected_intent == "ACCOUNT" for q in account_queries)

    def test_get_queries_filter_by_confidence(self, test_config: AppConfig):
        """Test filtering by confidence range."""
        db = DatabaseManager(test_config)
        db.clear_all()

        db.save_query("Q1", "R1", "OTHER", "A1", 30, "test.md")
        db.save_query("Q2", "R2", "OTHER", "A2", 50, "test.md")
        db.save_query("Q3", "R3", "OTHER", "A3", 80, "test.md")

        mid_confidence = db.get_queries(min_confidence=40, max_confidence=60)

        assert len(mid_confidence) == 1
        assert mid_confidence[0].confidence_score == 50

    def test_get_low_confidence_queries(self, test_config: AppConfig):
        """Test getting low confidence queries."""
        db = DatabaseManager(test_config)
        db.clear_all()

        db.save_query("Q1", "R1", "OTHER", "A1", 20, "test.md")  # Low
        db.save_query("Q2", "R2", "OTHER", "A2", 35, "test.md")  # Low
        db.save_query("Q3", "R3", "OTHER", "A3", 80, "test.md")  # High

        low_conf = db.get_low_confidence_queries(threshold=40)

        assert len(low_conf) == 2
        assert all(q.confidence_score < 40 for q in low_conf)

    def test_get_stats_empty(self, test_config: AppConfig):
        """Test stats on empty database."""
        db = DatabaseManager(test_config)
        db.clear_all()

        stats = db.get_stats()

        assert stats["total_queries"] == 0
        assert stats["avg_confidence"] == 0
        assert stats["avg_response_time_ms"] == 0

    def test_get_stats_with_data(self, test_config: AppConfig):
        """Test stats with data."""
        db = DatabaseManager(test_config)
        db.clear_all()

        db.save_query("Q1", "R1", "ACCOUNT", "A1", 60, "test.md", 100)
        db.save_query("Q2", "R2", "LOANS", "A2", 80, "test.md", 200)
        db.save_query("Q3", "R3", "ACCOUNT", "A3", 40, "test.md", 150)

        stats = db.get_stats()

        assert stats["total_queries"] == 3
        assert stats["avg_confidence"] == 60.0  # (60+80+40)/3
        assert stats["avg_response_time_ms"] == 150.0  # (100+200+150)/3
        assert stats["intent_distribution"]["ACCOUNT"] == 2
        assert stats["intent_distribution"]["LOANS"] == 1

    def test_get_stats_confidence_distribution(self, test_config: AppConfig):
        """Test confidence distribution in stats."""
        db = DatabaseManager(test_config)
        db.clear_all()

        # Low: <40
        db.save_query("Q1", "R1", "OTHER", "A1", 20, "test.md")
        db.save_query("Q2", "R2", "OTHER", "A2", 35, "test.md")

        # Medium: 40-69
        db.save_query("Q3", "R3", "OTHER", "A3", 50, "test.md")

        # High: >=70
        db.save_query("Q4", "R4", "OTHER", "A4", 75, "test.md")
        db.save_query("Q5", "R5", "OTHER", "A5", 90, "test.md")

        stats = db.get_stats()

        assert stats["confidence_distribution"]["low"] == 2
        assert stats["confidence_distribution"]["medium"] == 1
        assert stats["confidence_distribution"]["high"] == 2

    def test_document_usage_tracking(self, test_config: AppConfig):
        """Test document usage is tracked."""
        db = DatabaseManager(test_config)
        db.clear_all()

        db.save_query("Q1", "R1", "OTHER", "A1", 50, "doc1.md")
        db.save_query("Q2", "R2", "OTHER", "A2", 50, "doc1.md")
        db.save_query("Q3", "R3", "OTHER", "A3", 50, "doc2.md")

        usages = db.get_document_usage()

        assert len(usages) == 2

        # doc1.md should have 2 usages, doc2.md should have 1
        usage_dict = {u.document_name: u.usage_count for u in usages}
        assert usage_dict["doc1.md"] == 2
        assert usage_dict["doc2.md"] == 1

    def test_document_usage_sorted_by_count(self, test_config: AppConfig):
        """Test document usage is sorted by count."""
        db = DatabaseManager(test_config)
        db.clear_all()

        # Create varied usage
        for _ in range(3):
            db.save_query("Q", "R", "OTHER", "A", 50, "popular.md")
        for _ in range(1):
            db.save_query("Q", "R", "OTHER", "A", 50, "rare.md")

        usages = db.get_document_usage()

        assert usages[0].document_name == "popular.md"
        assert usages[0].usage_count == 3

    def test_document_usage_with_none_source(self, test_config: AppConfig):
        """Test that 'none' source documents are not tracked."""
        db = DatabaseManager(test_config)
        db.clear_all()

        db.save_query("Q1", "R1", "OTHER", "A1", 50, "none")
        db.save_query("Q2", "R2", "OTHER", "A2", 50, "real.md")

        usages = db.get_document_usage()

        assert len(usages) == 1
        assert usages[0].document_name == "real.md"

    def test_get_total_query_count(self, test_config: AppConfig):
        """Test getting total query count."""
        db = DatabaseManager(test_config)
        db.clear_all()

        assert db.get_total_query_count() == 0

        db.save_query("Q1", "R1", "OTHER", "A1", 50, "test.md")
        db.save_query("Q2", "R2", "OTHER", "A2", 50, "test.md")

        assert db.get_total_query_count() == 2

    def test_clear_all(self, test_config: AppConfig):
        """Test clearing all data."""
        db = DatabaseManager(test_config)

        db.save_query("Q1", "R1", "OTHER", "A1", 50, "test.md")
        assert db.get_total_query_count() >= 1

        db.clear_all()

        assert db.get_total_query_count() == 0
        assert len(db.get_document_usage()) == 0

    def test_queries_ordered_by_date(self, test_config: AppConfig):
        """Test that queries are ordered by date descending."""
        db = DatabaseManager(test_config)
        db.clear_all()

        db.save_query("Q1", "R1", "OTHER", "A1", 50, "test.md")
        db.save_query("Q2", "R2", "OTHER", "A2", 50, "test.md")
        db.save_query("Q3", "R3", "OTHER", "A3", 50, "test.md")

        queries = db.get_queries()

        # Most recent should be first
        assert queries[0].question == "Q3"
        assert queries[2].question == "Q1"
