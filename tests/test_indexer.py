"""Unit tests for the document indexer."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from app.rag.indexer import DocumentIndexer, Document, Chunk
from app.config import AppConfig


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            name="test.md",
            content="Test content",
            path="/path/to/test.md"
        )
        assert doc.name == "test.md"
        assert doc.content == "Test content"
        assert doc.path == "/path/to/test.md"


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a chunk."""
        chunk = Chunk(
            document_name="test.md",
            content="Chunk content",
            chunk_index=0
        )
        assert chunk.document_name == "test.md"
        assert chunk.content == "Chunk content"
        assert chunk.chunk_index == 0

    def test_chunk_id_generation(self):
        """Test chunk ID generation."""
        chunk = Chunk(
            document_name="test.md",
            content="Content",
            chunk_index=5
        )
        assert chunk.chunk_id == "test.md_5"

    def test_chunk_with_metadata(self):
        """Test chunk with metadata."""
        chunk = Chunk(
            document_name="test.md",
            content="Content",
            chunk_index=0,
            metadata={"source": "test.md", "section": "intro"}
        )
        assert chunk.metadata["source"] == "test.md"
        assert chunk.metadata["section"] == "intro"


class TestDocumentIndexer:
    """Tests for DocumentIndexer class."""

    def test_load_documents(self, sample_knowledge_base: Path, test_config: AppConfig):
        """Test loading documents from knowledge base."""
        with patch('ollama.Client'):
            indexer = DocumentIndexer(config=test_config)
            documents = indexer.load_documents(str(sample_knowledge_base))

            assert len(documents) == 2
            doc_names = {doc.name for doc in documents}
            assert "account_opening.md" in doc_names
            assert "fees_charges.md" in doc_names

    def test_load_documents_empty_directory(self, tmp_path: Path, test_config: AppConfig):
        """Test loading from empty directory."""
        empty_dir = tmp_path / "empty_kb"
        empty_dir.mkdir()

        with patch('ollama.Client'):
            indexer = DocumentIndexer(config=test_config)
            documents = indexer.load_documents(str(empty_dir))

            assert len(documents) == 0

    def test_load_documents_nonexistent_directory(self, tmp_path: Path, test_config: AppConfig):
        """Test loading from nonexistent directory."""
        with patch('ollama.Client'):
            indexer = DocumentIndexer(config=test_config)
            documents = indexer.load_documents(str(tmp_path / "nonexistent"))

            assert len(documents) == 0

    def test_chunk_document_basic(self, test_config: AppConfig):
        """Test basic document chunking."""
        with patch('ollama.Client'):
            indexer = DocumentIndexer(config=test_config)

            doc = Document(
                name="test.md",
                content="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
                path="/test.md"
            )

            chunks = indexer.chunk_document(doc, chunk_size=50, chunk_overlap=0)

            assert len(chunks) >= 1
            assert all(chunk.document_name == "test.md" for chunk in chunks)
            assert chunks[0].chunk_index == 0

    def test_chunk_document_empty(self, test_config: AppConfig):
        """Test chunking empty document."""
        with patch('ollama.Client'):
            indexer = DocumentIndexer(config=test_config)

            doc = Document(name="empty.md", content="", path="/empty.md")
            chunks = indexer.chunk_document(doc)

            assert len(chunks) == 0

    def test_chunk_document_whitespace_only(self, test_config: AppConfig):
        """Test chunking document with only whitespace."""
        with patch('ollama.Client'):
            indexer = DocumentIndexer(config=test_config)

            doc = Document(name="whitespace.md", content="   \n\n   ", path="/whitespace.md")
            chunks = indexer.chunk_document(doc)

            assert len(chunks) == 0

    def test_chunk_document_respects_size(self, test_config: AppConfig):
        """Test that chunks respect size limit."""
        with patch('ollama.Client'):
            indexer = DocumentIndexer(config=test_config)

            # Create a document with known content
            content = "Word " * 100  # 500 characters
            doc = Document(name="test.md", content=content, path="/test.md")

            chunks = indexer.chunk_document(doc, chunk_size=100, chunk_overlap=0)

            # Each chunk should be at or under the size limit (with some flexibility for word boundaries)
            for chunk in chunks:
                assert len(chunk.content) <= 150  # Allow some overage for word boundaries

    def test_chunk_document_preserves_content(self, test_config: AppConfig):
        """Test that chunking preserves all content."""
        with patch('ollama.Client'):
            indexer = DocumentIndexer(config=test_config)

            original_content = "Para one.\n\nPara two.\n\nPara three."
            doc = Document(name="test.md", content=original_content, path="/test.md")

            chunks = indexer.chunk_document(doc, chunk_size=1000, chunk_overlap=0)

            # With large chunk size, should fit in one chunk
            combined = "\n\n".join(chunk.content for chunk in chunks)
            assert "Para one" in combined
            assert "Para two" in combined
            assert "Para three" in combined

    def test_index_chunks_with_mock_ollama(self, test_config: AppConfig, mock_chroma_client):
        """Test indexing chunks with mocked Ollama."""
        mock_ollama = Mock()
        mock_ollama.embeddings.return_value = {"embedding": [0.1] * 768}

        indexer = DocumentIndexer(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        chunks = [
            Chunk(document_name="test.md", content="Content 1", chunk_index=0, metadata={"source": "test.md"}),
            Chunk(document_name="test.md", content="Content 2", chunk_index=1, metadata={"source": "test.md"}),
        ]

        count = indexer.index_chunks(chunks)

        assert count == 2
        assert mock_ollama.embeddings.call_count == 2
        assert indexer.get_indexed_count() == 2

    def test_index_chunks_empty_list(self, test_config: AppConfig, mock_chroma_client):
        """Test indexing empty chunk list."""
        mock_ollama = Mock()

        indexer = DocumentIndexer(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        count = indexer.index_chunks([])

        assert count == 0
        assert mock_ollama.embeddings.call_count == 0

    def test_embed_text(self, test_config: AppConfig, mock_chroma_client):
        """Test text embedding."""
        mock_ollama = Mock()
        expected_embedding = [0.1, 0.2, 0.3]
        mock_ollama.embeddings.return_value = {"embedding": expected_embedding}

        indexer = DocumentIndexer(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        embedding = indexer.embed_text("test query")

        assert embedding == expected_embedding
        mock_ollama.embeddings.assert_called_once()

    def test_clear_index(self, test_config: AppConfig, mock_chroma_client):
        """Test clearing the index."""
        mock_ollama = Mock()
        mock_ollama.embeddings.return_value = {"embedding": [0.1] * 768}

        indexer = DocumentIndexer(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        # Add some chunks
        chunks = [
            Chunk(document_name="test.md", content="Content", chunk_index=0, metadata={"source": "test.md"})
        ]
        indexer.index_chunks(chunks)
        assert indexer.get_indexed_count() == 1

        # Clear index
        indexer.clear_index()
        assert indexer.get_indexed_count() == 0

    def test_index_all(self, sample_knowledge_base: Path, test_config: AppConfig, mock_chroma_client):
        """Test indexing all documents."""
        mock_ollama = Mock()
        mock_ollama.embeddings.return_value = {"embedding": [0.1] * 768}

        indexer = DocumentIndexer(
            config=test_config,
            ollama_client=mock_ollama,
            chroma_client=mock_chroma_client
        )

        count = indexer.index_all(str(sample_knowledge_base))

        assert count > 0
        assert indexer.get_indexed_count() > 0

    def test_get_indexed_count_empty(self, test_config: AppConfig, mock_chroma_client):
        """Test getting count of empty index."""
        with patch('ollama.Client'):
            indexer = DocumentIndexer(
                config=test_config,
                chroma_client=mock_chroma_client
            )
            assert indexer.get_indexed_count() == 0
