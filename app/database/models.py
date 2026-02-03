"""SQLAlchemy models for the Customer Service Support System."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Query(Base):
    """Model for storing query history."""

    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    reformulated_query = Column(Text, nullable=False)
    detected_intent = Column(String(50), nullable=False, default="OTHER")
    answer = Column(Text, nullable=False)
    confidence_score = Column(Integer, nullable=False)
    source_document = Column(String(255), nullable=False)
    response_time_ms = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"<Query(id={self.id}, question='{self.question[:50]}...', confidence={self.confidence_score})>"

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "reformulated_query": self.reformulated_query,
            "detected_intent": self.detected_intent,
            "answer": self.answer,
            "confidence_score": self.confidence_score,
            "source_document": self.source_document,
            "response_time_ms": self.response_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DocumentUsage(Base):
    """Model for tracking document usage statistics."""

    __tablename__ = "document_usage"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_name = Column(String(255), nullable=False, unique=True)
    usage_count = Column(Integer, nullable=False, default=0)
    last_used = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"<DocumentUsage(document='{self.document_name}', count={self.usage_count})>"

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "document_name": self.document_name,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }
