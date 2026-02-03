"""Document retriever for RAG system - handles similarity search and retrieval."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import chromadb
import ollama

from app.config import AppConfig, get_config, KNOWLEDGE_BASE_DIR


@dataclass
class RetrievalResult:
    """Represents a retrieval result from the vector search."""
    document_name: str
    content: str
    score: float
    chunk_id: str

    @property
    def relevance_percentage(self) -> float:
        """Convert cosine distance to relevance percentage."""
        # ChromaDB returns distance (lower is better), convert to similarity
        return round(max(0, min(100, (1 - self.score) * 100)), 1)


class DocumentRetriever:
    """Retrieves relevant documents from ChromaDB based on query similarity."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        ollama_client: Optional[ollama.Client] = None,
        chroma_client: Optional[chromadb.Client] = None,
    ):
        """Initialize the document retriever.

        Args:
            config: Application configuration
            ollama_client: Ollama client for embeddings (optional, creates default)
            chroma_client: ChromaDB client (optional, creates default)
        """
        self.config = config or get_config()
        self.ollama_client = ollama_client or ollama.Client(host=self.config.ollama.host)

        # Initialize ChromaDB
        if chroma_client:
            self.chroma_client = chroma_client
        else:
            persist_dir = self.config.chroma.persist_directory
            if persist_dir and os.path.exists(persist_dir):
                self.chroma_client = chromadb.PersistentClient(path=persist_dir)
            else:
                self.chroma_client = chromadb.Client()

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.chroma.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        response = self.ollama_client.embeddings(
            model=self.config.ollama.embedding_model,
            prompt=query
        )
        return response["embedding"]

    def search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> list[RetrievalResult]:
        """Search for relevant documents based on query.

        Args:
            query: Search query
            top_k: Number of results to return (uses config default if None)

        Returns:
            List of retrieval results sorted by relevance
        """
        top_k = top_k or self.config.rag.top_k

        # Check if collection has any documents
        if self.collection.count() == 0:
            return []

        # Embed the query
        query_embedding = self.embed_query(query)

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Convert to RetrievalResult objects
        retrieval_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                retrieval_results.append(RetrievalResult(
                    document_name=results["metadatas"][0][i].get("source", "unknown"),
                    content=results["documents"][0][i],
                    score=results["distances"][0][i],
                    chunk_id=chunk_id
                ))

        return retrieval_results

    def get_document_content(
        self,
        doc_name: str,
        knowledge_base_path: Optional[str] = None
    ) -> Optional[str]:
        """Get the full content of a document by name.

        Args:
            doc_name: Name of the document (e.g., "account_opening.md")
            knowledge_base_path: Path to knowledge base directory

        Returns:
            Document content or None if not found
        """
        kb_path = Path(knowledge_base_path) if knowledge_base_path else KNOWLEDGE_BASE_DIR
        doc_path = kb_path / doc_name

        if doc_path.exists():
            return doc_path.read_text(encoding="utf-8")
        return None

    def get_document_names(self) -> list[str]:
        """Get list of all indexed document names.

        Returns:
            List of unique document names
        """
        # Get all metadata from collection
        results = self.collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return []

        # Extract unique document names
        doc_names = set()
        for metadata in results["metadatas"]:
            if "source" in metadata:
                doc_names.add(metadata["source"])

        return sorted(list(doc_names))

    def search_with_context(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> tuple[list[RetrievalResult], str]:
        """Search and return results with combined context.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Tuple of (results, combined_context_string)
        """
        results = self.search(query, top_k)

        if not results:
            return results, ""

        # Combine contexts from all results
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {result.document_name}]\n{result.content}"
            )

        combined_context = "\n\n---\n\n".join(context_parts)
        return results, combined_context
