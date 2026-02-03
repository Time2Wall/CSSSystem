# Customer Service Support System
Multi-Agent RAG Application – Claude Code Specification

## 1. Objective

Build a working multi-agent application using Claude Code that assists bank customer service representatives in answering customer questions accurately and efficiently.

The system must:
- Use Retrieval-Augmented Generation (RAG)
- Orchestrate three specialized agents
- Provide a usable UI for representatives
- Provide a manager dashboard with meaningful analytics

---

## 2. Functional Architecture

### 2.1 High-Level Flow

Representative Question  
→ Reformulation Agent  
→ Search Agent (RAG)  
→ Validation Agent  
→ Final Answer + Confidence + Source

---

## 3. Knowledge Base (Banking Domain)

### 3.1 Content Requirements

Create 5–10 domain documents covering at least the following topics:

- Account opening policies
- Loans and mortgage terms
- Fees, charges, and refunds
- Credit cards (types, limits, benefits)
- Branch hours and contact information
- Common mobile / website issues and troubleshooting

### 3.2 RAG Implementation

- Documents must be indexed and searchable
- The system must retrieve relevant passages
- Returned answers must include a source reference
- The full source document must be viewable in the UI

---

## 4. Multi-Agent System

### 4.1 Agent 1 — Reformulation Agent

**Responsibilities:**
- Receive the raw question from a representative
- Identify the underlying intent
- Rewrite the question as an optimized search query

**Example:**
- Input: "Customer is yelling that money was stolen from his card"
- Output: "Unauthorized credit card charge dispute process and refund policy"

---

### 4.2 Agent 2 — Search Agent

**Responsibilities:**
- Receive the reformulated query
- Search the Knowledge Base using RAG
- Return:
  - A concise answer
  - A source document reference

---

### 4.3 Agent 3 — Validation Agent

**Responsibilities:**
- Evaluate the answer from the Search Agent
- Assess accuracy and relevance
- Return a confidence score (0–100%)

---

## 5. User Interface

### 5.1 Screen 1 — Representative View

**Required Elements:**
- Question input field
- Displayed answer
- Confidence score (visual indicator: color, icon, or badge)
- Source reference (document name)
- Button or link to open the full source document

**Optional:**
- Display the reformulated query (in UI or logs)

---

### 5.2 Screen 2 — Manager Dashboard

**User Statistics:**
- Number of queries per representative
- Topics or categories of questions
- Response usage frequency

**System and Agent Statistics:**
- Average confidence scores
- Response times
- Most frequently used documents
- Low-confidence answers

---

## 6. Technical Requirements

- **Programming Language: Python**
- All logic must be implemented using Claude Code
- The application must run end-to-end and produce real outputs
- The UI must be clear and practical for daily use
- Visual polish is not required; usability is the priority

---

## 7. Technical Stack

### 7.1 Core Technologies

| Component | Technology | Notes |
|-----------|------------|-------|
| LLM | Ollama + Llama 3.2 3B | Local, no API costs |
| Embeddings | Ollama + nomic-embed-text | For RAG vector search |
| Vector Database | ChromaDB | Lightweight, no external setup |
| Persistence | SQLite | Query history and analytics |
| Backend | FastAPI | REST API with async support |
| Frontend | HTML/CSS/JS | Served by FastAPI |
| Agent Communication | Direct function calls | Simple orchestration |

### 7.1.1 Ollama Models

| Model | Purpose | Size | Min RAM |
|-------|---------|------|---------|
| `llama3.2:3b` | Agent reasoning (reformulation, search, validation) | ~2GB | 8GB |
| `nomic-embed-text` | Document/query embeddings for RAG | ~274MB | 4GB |

**Installation:**
```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 7.2 Authentication

- Single-user demo (no authentication required)

### 7.3 Deployment

- Local only
- Requires Ollama running locally (`ollama serve`)
- No API keys needed

### 7.4 Knowledge Base Format

- Markdown files (`.md`)
- Fictional/generic banking content
- Stored in `/knowledge_base` directory

### 7.5 Project Structure

```
CSSSystem/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Configuration and settings
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── reformulation.py    # Agent 1: Query rewriting
│   │   ├── search.py           # Agent 2: RAG search
│   │   └── validation.py       # Agent 3: Confidence scoring
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── indexer.py          # Document indexing to ChromaDB
│   │   └── retriever.py        # Vector similarity search
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py           # SQLAlchemy models
│   │   └── db.py               # Database operations
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   └── templates/
│       ├── index.html          # Representative view
│       └── dashboard.html      # Manager dashboard
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── test_reformulation.py   # Unit: Reformulation Agent
│   ├── test_search.py          # Unit: Search Agent
│   ├── test_validation.py      # Unit: Validation Agent
│   ├── test_indexer.py         # Unit: RAG Indexer
│   ├── test_retriever.py       # Unit: RAG Retriever
│   ├── test_database.py        # Unit: Database operations
│   ├── test_agent_pipeline.py  # Integration: Agent chain
│   ├── test_api_endpoints.py   # Integration: API endpoints
│   └── test_end_to_end.py      # Integration: Full system
├── knowledge_base/             # 5-10 markdown banking docs
│   ├── account_opening.md
│   ├── loans_mortgages.md
│   ├── fees_charges.md
│   ├── credit_cards.md
│   ├── branch_info.md
│   └── mobile_troubleshooting.md
├── data/
│   └── css_system.db           # SQLite database
├── requirements.txt
├── pytest.ini                  # Pytest configuration
└── README.md
```

### 7.6 Dependencies

```
ollama              # Ollama Python client
fastapi             # Web framework
uvicorn             # ASGI server
chromadb            # Vector database
sqlalchemy          # ORM for SQLite
python-dotenv       # Environment variables
jinja2              # HTML templating
aiofiles            # Async file operations
pytest              # Testing framework
pytest-asyncio      # Async test support
httpx               # Test client for FastAPI
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

Each component must have unit tests covering core functionality:

| Component | Test File | What to Test |
|-----------|-----------|--------------|
| Reformulation Agent | `tests/test_reformulation.py` | Query rewriting, intent detection |
| Search Agent | `tests/test_search.py` | RAG retrieval, answer generation |
| Validation Agent | `tests/test_validation.py` | Confidence scoring logic |
| RAG Indexer | `tests/test_indexer.py` | Document parsing, embedding generation |
| RAG Retriever | `tests/test_retriever.py` | Vector search, relevance ranking |
| Database | `tests/test_database.py` | CRUD operations, analytics queries |

### 8.2 Integration Tests

| Test File | What to Test |
|-----------|--------------|
| `tests/test_agent_pipeline.py` | Full agent chain (Reformulation → Search → Validation) |
| `tests/test_api_endpoints.py` | All FastAPI endpoints |
| `tests/test_end_to_end.py` | Complete user flow from question to answer |

### 8.3 Test Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_reformulation.py

# Run only unit tests
pytest tests/ -k "not integration"

# Run only integration tests
pytest tests/ -k "integration"
```

---

## 9. Implementation Checklist

Track progress by checking off completed items. **All tests must pass before checking the box.**

### 9.1 Foundation
- [x] Project structure created
- [x] Dependencies installed (`requirements.txt`)
- [ ] Ollama models pulled and working
- [x] Configuration module (`config.py`)

### 9.2 Knowledge Base
- [x] Banking documents created (5-10 markdown files)
- [x] RAG Indexer implemented
- [x] RAG Retriever implemented
- [ ] Unit tests passing: `pytest tests/test_indexer.py tests/test_retriever.py`

### 9.3 Agents
- [x] Reformulation Agent implemented
- [ ] Unit tests passing: `pytest tests/test_reformulation.py`
- [x] Search Agent implemented
- [ ] Unit tests passing: `pytest tests/test_search.py`
- [x] Validation Agent implemented
- [ ] Unit tests passing: `pytest tests/test_validation.py`
- [ ] Agent pipeline integration test passing: `pytest tests/test_agent_pipeline.py`

### 9.4 Database
- [x] SQLite models defined
- [x] Database operations implemented
- [ ] Unit tests passing: `pytest tests/test_database.py`

### 9.5 API
- [x] FastAPI endpoints implemented
- [ ] API integration tests passing: `pytest tests/test_api_endpoints.py`

### 9.6 User Interface
- [x] Representative View (index.html) implemented
- [x] Manager Dashboard (dashboard.html) implemented
- [x] UI connected to API endpoints

### 9.7 Final Validation
- [ ] All unit tests passing: `pytest tests/ -k "not integration"`
- [ ] All integration tests passing: `pytest tests/ -k "integration"`
- [ ] End-to-end test passing: `pytest tests/test_end_to_end.py`
- [ ] Manual smoke test: full user flow works in browser
- [ ] System runs without errors: `uvicorn app.main:app`

---

## 10. Deliverables

1. Source code (GitHub repository or ZIP archive)
2. Instructions on how to run the application
3. Demo including:
   - A question being asked
   - The reformulated query (visible in UI or logs)
   - Answer with confidence score
   - Opening the source document
   - Manager dashboard with statistics

---

## 11. Evaluation Criteria

| Area | What Is Evaluated |
|-----|-------------------|
| Multi-Agent Design | Correct orchestration of all three agents |
| RAG Quality | Relevant, grounded answers |
| Claude Code Usage | Entire system built via Claude Code |
| UX | Usable by non-technical representatives |
| Manager Dashboard | Actionable and meaningful metrics |
| Test Coverage | All unit and integration tests passing |

---

## 12. Success Criteria

The system is successful if:
- All agents execute in the correct sequence
- Answers are grounded in retrieved documents
- Confidence scoring is meaningful
- Both UI screens are fully functional
- The application is suitable for real customer support usage
- All unit tests pass
- All integration tests pass
- End-to-end system test passes
