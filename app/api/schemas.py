"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for submitting a question."""
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")


class RelevantPassage(BaseModel):
    """Schema for a relevant passage from the knowledge base."""
    document_name: str
    content: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response schema for a processed query."""
    id: int
    question: str
    reformulated_query: str
    detected_intent: str
    answer: str
    confidence_score: int = Field(..., ge=0, le=100)
    confidence_level: str
    source_document: str
    relevant_passages: list[RelevantPassage] = []
    response_time_ms: int
    created_at: str


class QueryListItem(BaseModel):
    """Schema for a query in list view."""
    id: int
    question: str
    answer: str
    confidence_score: int
    confidence_level: str
    source_document: str
    detected_intent: str
    response_time_ms: int
    created_at: str


class QueryListResponse(BaseModel):
    """Response schema for query list."""
    queries: list[QueryListItem]
    total: int
    limit: int
    offset: int


class DocumentInfo(BaseModel):
    """Schema for document information."""
    name: str
    usage_count: int
    last_used: Optional[str] = None


class DocumentResponse(BaseModel):
    """Response schema for document content."""
    name: str
    content: str


class DocumentListResponse(BaseModel):
    """Response schema for document list."""
    documents: list[DocumentInfo]


class ConfidenceDistribution(BaseModel):
    """Schema for confidence distribution."""
    high: int
    medium: int
    low: int


class StatsResponse(BaseModel):
    """Response schema for dashboard statistics."""
    total_queries: int
    avg_confidence: float
    avg_response_time_ms: float
    confidence_distribution: ConfidenceDistribution
    intent_distribution: dict[str, int]
    queries_per_day: dict[str, int]
    top_documents: list[DocumentInfo]
    low_confidence_count: int


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    version: str
    indexed_documents: int
