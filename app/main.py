"""FastAPI application entry point for the Customer Service Support System."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
import ollama

from app.config import AppConfig, get_config, STATIC_DIR, TEMPLATES_DIR, KNOWLEDGE_BASE_DIR
from app.api.routes import router as api_router, set_dependencies
from app.agents.pipeline import AgentPipeline
from app.database.db import DatabaseManager
from app.rag.indexer import DocumentIndexer
from app.rag.retriever import DocumentRetriever
from app import __version__


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Application configuration (uses default if None)

    Returns:
        Configured FastAPI application
    """
    config = config or get_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan handler."""
        # Startup
        print(f"Starting Customer Service Support System v{__version__}")

        # Initialize Ollama client
        ollama_client = ollama.Client(host=config.ollama.host)

        # Initialize database
        db_manager = DatabaseManager(config)
        print("Database initialized")

        # Initialize RAG system
        indexer = DocumentIndexer(config=config, ollama_client=ollama_client)
        retriever = DocumentRetriever(config=config, ollama_client=ollama_client)

        # Check if we need to index documents
        if indexer.get_indexed_count() == 0:
            print("Indexing knowledge base documents...")
            try:
                count = indexer.index_all(str(KNOWLEDGE_BASE_DIR))
                print(f"Indexed {count} chunks from knowledge base")
            except Exception as e:
                print(f"Warning: Could not index documents: {e}")
                print("The system will start but RAG search may not work.")
        else:
            print(f"Knowledge base already indexed ({indexer.get_indexed_count()} chunks)")

        # Initialize agent pipeline
        pipeline = AgentPipeline(
            config=config,
            ollama_client=ollama_client,
            retriever=retriever
        )
        print("Agent pipeline initialized")

        # Set dependencies for API routes
        set_dependencies(pipeline, db_manager, retriever, config)

        # Store in app state for access
        app.state.config = config
        app.state.db_manager = db_manager
        app.state.pipeline = pipeline
        app.state.retriever = retriever
        app.state.indexer = indexer

        yield

        # Shutdown
        print("Shutting down...")

    app = FastAPI(
        title="Customer Service Support System",
        description="Multi-Agent RAG Application for Bank Customer Service",
        version=__version__,
        lifespan=lifespan
    )

    # Include API router
    app.include_router(api_router, prefix="/api")

    # Setup templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Mount static files if directory exists
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Serve the representative view."""
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Serve the manager dashboard."""
        return templates.TemplateResponse("dashboard.html", {"request": request})

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
