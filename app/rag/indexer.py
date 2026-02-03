"""Document indexer for RAG system - handles loading, chunking, and indexing documents."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import chromadb
import ollama

from app.config import AppConfig, get_config


@dataclass
class Document:
    """Represents a loaded document."""
    name: str
    content: str
    path: str


@dataclass
class Chunk:
    """Represents a chunk of a document."""
    document_name: str
    content: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Generate a unique ID for this chunk."""
        return f"{self.document_name}_{self.chunk_index}"


class DocumentIndexer:
    """Indexes documents into ChromaDB for vector search."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        ollama_client: Optional[ollama.Client] = None,
        chroma_client: Optional[chromadb.Client] = None,
    ):
        """Initialize the document indexer.

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
            if persist_dir:
                os.makedirs(persist_dir, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(path=persist_dir)
            else:
                self.chroma_client = chromadb.Client()

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.chroma.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def load_documents(self, knowledge_base_path: Optional[str] = None) -> list[Document]:
        """Load all markdown documents from the knowledge base directory.

        Args:
            knowledge_base_path: Path to knowledge base directory (uses config default if None)

        Returns:
            List of loaded documents
        """
        from app.config import KNOWLEDGE_BASE_DIR

        kb_path = Path(knowledge_base_path) if knowledge_base_path else KNOWLEDGE_BASE_DIR
        documents = []

        if not kb_path.exists():
            return documents

        for md_file in kb_path.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            documents.append(Document(
                name=md_file.name,
                content=content,
                path=str(md_file)
            ))

        return documents

    def chunk_document(
        self,
        doc: Document,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> list[Chunk]:
        """Split a document into chunks for indexing.

        Uses a simple sliding window approach with some awareness of paragraph breaks.

        Args:
            doc: Document to chunk
            chunk_size: Maximum characters per chunk (uses config default if None)
            chunk_overlap: Overlap between chunks (uses config default if None)

        Returns:
            List of chunks
        """
        chunk_size = chunk_size or self.config.rag.chunk_size
        chunk_overlap = chunk_overlap or self.config.rag.chunk_overlap

        content = doc.content.strip()
        if not content:
            return []

        # Split by paragraphs first (double newlines)
        paragraphs = re.split(r'\n\n+', content)
        chunks = []
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) + 2 > chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(Chunk(
                        document_name=doc.name,
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        metadata={"source": doc.name}
                    ))
                    chunk_index += 1

                    # Start new chunk with overlap
                    if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                        # Get last chunk_overlap characters
                        overlap_text = current_chunk[-chunk_overlap:]
                        # Try to start at a word boundary
                        space_idx = overlap_text.find(' ')
                        if space_idx > 0:
                            overlap_text = overlap_text[space_idx + 1:]
                        current_chunk = overlap_text + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    # Paragraph itself is too long, need to split it
                    words = para.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 > chunk_size:
                            if current_chunk:
                                chunks.append(Chunk(
                                    document_name=doc.name,
                                    content=current_chunk.strip(),
                                    chunk_index=chunk_index,
                                    metadata={"source": doc.name}
                                ))
                                chunk_index += 1
                            current_chunk = word
                        else:
                            current_chunk = f"{current_chunk} {word}" if current_chunk else word
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk = f"{current_chunk}\n\n{para}"
                else:
                    current_chunk = para

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(Chunk(
                document_name=doc.name,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                metadata={"source": doc.name}
            ))

        return chunks

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a text using Ollama.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = self.ollama_client.embeddings(
            model=self.config.ollama.embedding_model,
            prompt=text
        )
        return response["embedding"]

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """Index chunks into ChromaDB.

        Args:
            chunks: List of chunks to index

        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            metadatas.append(chunk.metadata)
            embeddings.append(self.embed_text(chunk.content))

        # Upsert to handle re-indexing
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return len(chunks)

    def index_all(self, knowledge_base_path: Optional[str] = None) -> int:
        """Index all documents from the knowledge base.

        Args:
            knowledge_base_path: Path to knowledge base directory

        Returns:
            Total number of chunks indexed
        """
        documents = self.load_documents(knowledge_base_path)
        total_chunks = 0

        for doc in documents:
            chunks = self.chunk_document(doc)
            total_chunks += self.index_chunks(chunks)

        return total_chunks

    def clear_index(self):
        """Clear all documents from the index."""
        # Delete and recreate collection
        self.chroma_client.delete_collection(self.config.chroma.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=self.config.chroma.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def get_indexed_count(self) -> int:
        """Get the number of indexed chunks."""
        return self.collection.count()
