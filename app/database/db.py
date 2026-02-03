"""Database operations for the Customer Service Support System."""

from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker, Session

from app.config import AppConfig, get_config
from app.database.models import Base, Query, DocumentUsage


class DatabaseManager:
    """Manages database operations for the CSS System."""

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize the database manager.

        Args:
            config: Application configuration
        """
        self.config = config or get_config()
        self.engine = create_engine(
            self.config.database.url,
            echo=self.config.database.echo
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def save_query(
        self,
        question: str,
        reformulated_query: str,
        detected_intent: str,
        answer: str,
        confidence_score: int,
        source_document: str,
        response_time_ms: int = 0,
    ) -> Query:
        """Save a query to the database.

        Args:
            question: Original question
            reformulated_query: Reformulated query
            detected_intent: Detected intent category
            answer: Generated answer
            confidence_score: Confidence score (0-100)
            source_document: Source document name
            response_time_ms: Response time in milliseconds

        Returns:
            The saved Query object
        """
        session = self.get_session()
        try:
            query = Query(
                question=question,
                reformulated_query=reformulated_query,
                detected_intent=detected_intent,
                answer=answer,
                confidence_score=confidence_score,
                source_document=source_document,
                response_time_ms=response_time_ms,
                created_at=datetime.utcnow()
            )
            session.add(query)

            # Update document usage
            self._update_document_usage(session, source_document)

            session.commit()
            session.refresh(query)
            return query
        finally:
            session.close()

    def _update_document_usage(self, session: Session, document_name: str):
        """Update document usage statistics.

        Args:
            session: Database session
            document_name: Name of the document used
        """
        if not document_name or document_name == "none":
            return

        doc_usage = session.query(DocumentUsage).filter(
            DocumentUsage.document_name == document_name
        ).first()

        if doc_usage:
            doc_usage.usage_count += 1
            doc_usage.last_used = datetime.utcnow()
        else:
            doc_usage = DocumentUsage(
                document_name=document_name,
                usage_count=1,
                last_used=datetime.utcnow()
            )
            session.add(doc_usage)

    def get_query(self, query_id: int) -> Optional[Query]:
        """Get a specific query by ID.

        Args:
            query_id: Query ID

        Returns:
            Query object or None if not found
        """
        session = self.get_session()
        try:
            return session.query(Query).filter(Query.id == query_id).first()
        finally:
            session.close()

    def get_queries(
        self,
        limit: int = 50,
        offset: int = 0,
        intent_filter: Optional[str] = None,
        min_confidence: Optional[int] = None,
        max_confidence: Optional[int] = None,
    ) -> list[Query]:
        """Get queries with optional filtering.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            intent_filter: Filter by intent category
            min_confidence: Minimum confidence score
            max_confidence: Maximum confidence score

        Returns:
            List of Query objects
        """
        session = self.get_session()
        try:
            query = session.query(Query)

            if intent_filter:
                query = query.filter(Query.detected_intent == intent_filter)
            if min_confidence is not None:
                query = query.filter(Query.confidence_score >= min_confidence)
            if max_confidence is not None:
                query = query.filter(Query.confidence_score <= max_confidence)

            query = query.order_by(desc(Query.created_at))
            query = query.offset(offset).limit(limit)

            return query.all()
        finally:
            session.close()

    def get_low_confidence_queries(
        self,
        threshold: Optional[int] = None,
        limit: int = 20
    ) -> list[Query]:
        """Get queries with low confidence scores.

        Args:
            threshold: Confidence threshold (uses config default if None)
            limit: Maximum number of results

        Returns:
            List of low-confidence Query objects
        """
        threshold = threshold or self.config.low_confidence_threshold
        return self.get_queries(limit=limit, max_confidence=threshold)

    def get_stats(self) -> dict:
        """Get aggregated statistics.

        Returns:
            Dictionary with various statistics
        """
        session = self.get_session()
        try:
            # Total queries
            total_queries = session.query(func.count(Query.id)).scalar() or 0

            # Average confidence
            avg_confidence = session.query(func.avg(Query.confidence_score)).scalar()
            avg_confidence = round(avg_confidence, 1) if avg_confidence else 0

            # Average response time
            avg_response_time = session.query(func.avg(Query.response_time_ms)).scalar()
            avg_response_time = round(avg_response_time, 1) if avg_response_time else 0

            # Confidence distribution
            high_confidence = session.query(func.count(Query.id)).filter(
                Query.confidence_score >= self.config.high_confidence_threshold
            ).scalar() or 0

            medium_confidence = session.query(func.count(Query.id)).filter(
                Query.confidence_score >= self.config.low_confidence_threshold,
                Query.confidence_score < self.config.high_confidence_threshold
            ).scalar() or 0

            low_confidence = session.query(func.count(Query.id)).filter(
                Query.confidence_score < self.config.low_confidence_threshold
            ).scalar() or 0

            # Intent distribution
            intent_counts = session.query(
                Query.detected_intent,
                func.count(Query.id)
            ).group_by(Query.detected_intent).all()

            intent_distribution = {intent: count for intent, count in intent_counts}

            # Queries per day (last 7 days)
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            daily_counts = session.query(
                func.date(Query.created_at),
                func.count(Query.id)
            ).filter(
                Query.created_at >= seven_days_ago
            ).group_by(
                func.date(Query.created_at)
            ).all()

            queries_per_day = {str(date): count for date, count in daily_counts}

            return {
                "total_queries": total_queries,
                "avg_confidence": avg_confidence,
                "avg_response_time_ms": avg_response_time,
                "confidence_distribution": {
                    "high": high_confidence,
                    "medium": medium_confidence,
                    "low": low_confidence
                },
                "intent_distribution": intent_distribution,
                "queries_per_day": queries_per_day
            }
        finally:
            session.close()

    def get_document_usage(self, limit: int = 10) -> list[DocumentUsage]:
        """Get document usage statistics.

        Args:
            limit: Maximum number of results

        Returns:
            List of DocumentUsage objects sorted by usage count
        """
        session = self.get_session()
        try:
            return session.query(DocumentUsage).order_by(
                desc(DocumentUsage.usage_count)
            ).limit(limit).all()
        finally:
            session.close()

    def get_total_query_count(self) -> int:
        """Get total number of queries.

        Returns:
            Total query count
        """
        session = self.get_session()
        try:
            return session.query(func.count(Query.id)).scalar() or 0
        finally:
            session.close()

    def clear_all(self):
        """Clear all data from the database. Use with caution!"""
        session = self.get_session()
        try:
            session.query(Query).delete()
            session.query(DocumentUsage).delete()
            session.commit()
        finally:
            session.close()


# Global instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager(config: Optional[AppConfig] = None) -> DatabaseManager:
    """Get the global database manager instance.

    Args:
        config: Application configuration (only used for first initialization)

    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(config)
    return _db_manager


def reset_db_manager():
    """Reset the global database manager (for testing)."""
    global _db_manager
    _db_manager = None
