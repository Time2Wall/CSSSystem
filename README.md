# Customer Service Support System

A Multi-Agent RAG Application for Bank Customer Service using Ollama (local LLMs), ChromaDB, SQLite, and FastAPI.

## Overview

This system assists bank customer service representatives in answering customer questions accurately and efficiently. It uses:

- **Retrieval-Augmented Generation (RAG)** to find relevant information from a knowledge base
- **Three specialized agents** that work together to process questions
- **Local LLMs via Ollama** for privacy and cost-effectiveness
- **A clean web interface** for representatives and managers

## Architecture

### Agent Pipeline

1. **Reformulation Agent**: Takes raw questions (often emotional or incomplete) and converts them into optimized search queries
2. **Search Agent**: Uses RAG to find relevant information and generates a grounded answer
3. **Validation Agent**: Evaluates answer quality and assigns a confidence score (0-100%)

### Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Ollama + Llama 3.2 3B |
| Embeddings | Ollama + nomic-embed-text |
| Vector Database | ChromaDB |
| Persistence | SQLite |
| Backend | FastAPI |
| Frontend | HTML/CSS/JS |

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

## Installation

### 1. Install Ollama and Models

```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Start Ollama server (if not running)
ollama serve
```

### 2. Clone and Set Up the Project

```bash
# Clone the repository
cd CSSSystem

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application

```bash
# Start the application
uvicorn app.main:app --reload

# Or run directly
python -m app.main
```

The application will be available at:
- **Representative View**: http://localhost:8000
- **Manager Dashboard**: http://localhost:8000/dashboard
- **API Documentation**: http://localhost:8000/docs

## Usage

### Representative View

1. Open http://localhost:8000
2. Enter a customer question in the text area
3. Click "Get Answer"
4. View the answer with confidence score
5. Click the source document link to see the full source

### Manager Dashboard

1. Open http://localhost:8000/dashboard
2. View statistics:
   - Total queries
   - Average confidence scores
   - Response times
   - Most used documents
   - Low-confidence queries needing attention

## Project Structure

```
CSSSystem/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Configuration settings
│   ├── agents/                 # Multi-agent system
│   │   ├── reformulation.py    # Query rewriting
│   │   ├── search.py           # RAG search
│   │   ├── validation.py       # Confidence scoring
│   │   └── pipeline.py         # Agent orchestration
│   ├── rag/                    # RAG system
│   │   ├── indexer.py          # Document indexing
│   │   └── retriever.py        # Vector search
│   ├── database/               # Data persistence
│   │   ├── models.py           # SQLAlchemy models
│   │   └── db.py               # Database operations
│   ├── api/                    # REST API
│   │   ├── routes.py           # API endpoints
│   │   └── schemas.py          # Pydantic models
│   ├── static/                 # CSS/JS files
│   └── templates/              # HTML templates
├── knowledge_base/             # Banking documents
├── tests/                      # Test suite
├── data/                       # SQLite database
└── requirements.txt
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Submit a question |
| `/api/query/{id}` | GET | Get specific query |
| `/api/queries` | GET | Query history |
| `/api/documents` | GET | List documents |
| `/api/documents/{name}` | GET | Get document content |
| `/api/stats` | GET | Dashboard statistics |
| `/api/health` | GET | Health check |

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run only unit tests
pytest tests/ -k "not integration"

# Run only integration tests
pytest tests/ -k "integration"

# Run specific test file
pytest tests/test_reformulation.py
```

## Configuration

Configuration is handled in `app/config.py`. Key settings:

```python
# Ollama settings
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

# RAG settings
CHUNK_SIZE = 500
TOP_K = 3

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 70
LOW_CONFIDENCE_THRESHOLD = 40
```

Environment variables can override defaults:
- `OLLAMA_HOST`
- `OLLAMA_LLM_MODEL`
- `OLLAMA_EMBEDDING_MODEL`
- `DATABASE_URL`
- `DEBUG`

## Knowledge Base

The knowledge base contains markdown files covering:
- Account opening policies
- Loans and mortgages
- Fees and charges
- Credit cards
- Branch information
- Mobile app troubleshooting

To add new documents:
1. Create a markdown file in `knowledge_base/`
2. Restart the application (documents are indexed on startup)

## Development

### Adding New Documents

1. Create `.md` file in `knowledge_base/`
2. Restart the app to re-index

### Modifying Agents

Each agent has its own prompt in its respective file:
- `app/agents/reformulation.py` - REFORMULATION_SYSTEM_PROMPT
- `app/agents/search.py` - SEARCH_SYSTEM_PROMPT
- `app/agents/validation.py` - VALIDATION_SYSTEM_PROMPT

### Database Migrations

Using SQLAlchemy models - tables are created automatically on startup.

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### Model Not Found

```bash
# List available models
ollama list

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Slow Responses

- Ensure you have adequate RAM (8GB minimum)
- Consider using a GPU for faster inference
- Check Ollama logs for issues

## License

This project is for educational purposes.
