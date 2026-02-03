"""API routes for the Customer Service Support System."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    QueryListResponse,
    QueryListItem,
    StatsResponse,
    DocumentResponse,
    DocumentListResponse,
    DocumentInfo,
    ConfidenceDistribution,
    HealthResponse,
    RelevantPassage,
)
from app.agents.pipeline import AgentPipeline
from app.database.db import DatabaseManager, get_db_manager
from app.rag.retriever import DocumentRetriever
from app.config import AppConfig, get_config, KNOWLEDGE_BASE_DIR
from app import __version__

router = APIRouter()

# Global instances (initialized in main.py)
_pipeline: Optional[AgentPipeline] = None
_db_manager: Optional[DatabaseManager] = None
_retriever: Optional[DocumentRetriever] = None
_config: Optional[AppConfig] = None


def set_dependencies(
    pipeline: AgentPipeline,
    db_manager: DatabaseManager,
    retriever: DocumentRetriever,
    config: AppConfig
):
    """Set the global dependencies for the API routes."""
    global _pipeline, _db_manager, _retriever, _config
    _pipeline = pipeline
    _db_manager = db_manager
    _retriever = retriever
    _config = config


def get_pipeline() -> AgentPipeline:
    """Get the agent pipeline."""
    if _pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    return _pipeline


def get_db() -> DatabaseManager:
    """Get the database manager."""
    if _db_manager is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return _db_manager


def get_retriever() -> DocumentRetriever:
    """Get the document retriever."""
    if _retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    return _retriever


def get_app_config() -> AppConfig:
    """Get the app config."""
    if _config is None:
        return get_config()
    return _config


def get_confidence_level(score: int) -> str:
    """Get confidence level from score."""
    config = get_app_config()
    if score >= config.high_confidence_threshold:
        return "high"
    elif score >= config.low_confidence_threshold:
        return "medium"
    else:
        return "low"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    retriever = get_retriever()
    indexed_count = retriever.collection.count() if retriever else 0

    return HealthResponse(
        status="healthy",
        version=__version__,
        indexed_documents=indexed_count
    )


@router.post("/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest):
    """Submit a question and get an answer."""
    pipeline = get_pipeline()
    db = get_db()

    # Process the question through the agent pipeline
    result = await pipeline.process(request.question)

    # Save to database
    query = db.save_query(
        question=result.original_question,
        reformulated_query=result.reformulated_query,
        detected_intent=result.detected_intent,
        answer=result.answer,
        confidence_score=result.confidence_score,
        source_document=result.source_document,
        response_time_ms=result.total_time_ms
    )

    # Build response
    relevant_passages = [
        RelevantPassage(
            document_name=p.document_name,
            content=p.content,
            relevance_score=p.relevance_percentage
        )
        for p in result.relevant_passages
    ]

    return QueryResponse(
        id=query.id,
        question=result.original_question,
        reformulated_query=result.reformulated_query,
        detected_intent=result.detected_intent,
        answer=result.answer,
        confidence_score=result.confidence_score,
        confidence_level=result.confidence_level,
        source_document=result.source_document,
        relevant_passages=relevant_passages,
        response_time_ms=result.total_time_ms,
        created_at=query.created_at.isoformat()
    )


@router.get("/query/{query_id}", response_model=QueryResponse)
async def get_query(query_id: int):
    """Get a specific query by ID."""
    db = get_db()

    query = db.get_query(query_id)
    if not query:
        raise HTTPException(status_code=404, detail="Query not found")

    return QueryResponse(
        id=query.id,
        question=query.question,
        reformulated_query=query.reformulated_query,
        detected_intent=query.detected_intent,
        answer=query.answer,
        confidence_score=query.confidence_score,
        confidence_level=get_confidence_level(query.confidence_score),
        source_document=query.source_document,
        relevant_passages=[],  # Not stored in DB
        response_time_ms=query.response_time_ms,
        created_at=query.created_at.isoformat()
    )


@router.get("/queries", response_model=QueryListResponse)
async def get_queries(
    limit: int = 50,
    offset: int = 0,
    intent: Optional[str] = None,
    min_confidence: Optional[int] = None,
    max_confidence: Optional[int] = None
):
    """Get query history with optional filtering."""
    db = get_db()

    queries = db.get_queries(
        limit=limit,
        offset=offset,
        intent_filter=intent,
        min_confidence=min_confidence,
        max_confidence=max_confidence
    )

    total = db.get_total_query_count()

    items = [
        QueryListItem(
            id=q.id,
            question=q.question,
            answer=q.answer[:200] + "..." if len(q.answer) > 200 else q.answer,
            confidence_score=q.confidence_score,
            confidence_level=get_confidence_level(q.confidence_score),
            source_document=q.source_document,
            detected_intent=q.detected_intent,
            response_time_ms=q.response_time_ms,
            created_at=q.created_at.isoformat()
        )
        for q in queries
    ]

    return QueryListResponse(
        queries=items,
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/documents", response_model=DocumentListResponse)
async def get_documents():
    """Get list of all knowledge base documents."""
    retriever = get_retriever()
    db = get_db()

    # Get document names from retriever
    doc_names = retriever.get_document_names()

    # Get usage stats
    usages = db.get_document_usage(limit=100)
    usage_dict = {u.document_name: u for u in usages}

    documents = []
    for name in doc_names:
        usage = usage_dict.get(name)
        documents.append(DocumentInfo(
            name=name,
            usage_count=usage.usage_count if usage else 0,
            last_used=usage.last_used.isoformat() if usage and usage.last_used else None
        ))

    # Sort by usage count
    documents.sort(key=lambda d: d.usage_count, reverse=True)

    return DocumentListResponse(documents=documents)


@router.get("/documents/{doc_name}", response_model=DocumentResponse)
async def get_document(doc_name: str):
    """Get full content of a document."""
    retriever = get_retriever()

    content = retriever.get_document_content(doc_name)
    if content is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(
        name=doc_name,
        content=content
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get dashboard statistics."""
    db = get_db()
    config = get_app_config()

    stats = db.get_stats()
    top_docs = db.get_document_usage(limit=5)
    low_conf_queries = db.get_low_confidence_queries(limit=100)

    return StatsResponse(
        total_queries=stats["total_queries"],
        avg_confidence=stats["avg_confidence"],
        avg_response_time_ms=stats["avg_response_time_ms"],
        confidence_distribution=ConfidenceDistribution(
            high=stats["confidence_distribution"]["high"],
            medium=stats["confidence_distribution"]["medium"],
            low=stats["confidence_distribution"]["low"]
        ),
        intent_distribution=stats["intent_distribution"],
        queries_per_day=stats["queries_per_day"],
        top_documents=[
            DocumentInfo(
                name=d.document_name,
                usage_count=d.usage_count,
                last_used=d.last_used.isoformat() if d.last_used else None
            )
            for d in top_docs
        ],
        low_confidence_count=len(low_conf_queries)
    )
