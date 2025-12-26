# Medical Transplant RAG System 🏥

**Production-ready Retrieval-Augmented Generation system for clinical transplant medicine**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Architecture](#️-architecture)
- [API Reference](#-api-reference)
- [Benchmarks](#-benchmarks)
- [Project Structure](#-project-structure)
- [Configuration](#️-configuration)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

A **medical-grade RAG system** that answers questions about organ transplantation using:
- **194 indexed chunks** from 24 medical documents
- **GPU-accelerated** embeddings and generation (RTX 3050 4GB VRAM)
- **Confidence scoring** for answer reliability (High/Medium/Low)
- **Source attribution** with document + section citations
- **REST API** with automatic documentation
- **Query logging** for audit trail and monitoring
- **Input validation** for medical safety

**Status:** ✅ Production Ready | **Version:** 1.0.0 | **Last Updated:** December 27, 2025

---

## ✨ Key Features

### 🧠 Intelligent RAG Pipeline
- Semantic retrieval with all-mpnet-base-v2 embeddings
- Token-aware deduplication and context budgeting
- Temperature-optimized generation (0.05 for clinical accuracy)
- Institutional variability hedging in responses

### 🔒 Production Safety
- **Confidence scoring**: High (>0.70) / Medium (0.55-0.70) / Low (<0.55)
- **Input validation**: Length checks, prohibited terms blocking
- **Query logging**: JSONL audit trail with timestamps
- **Error handling**: Proper HTTP status codes

### 📡 REST API
- FastAPI with auto-generated Swagger UI
- `/api/v1/query`: Answer medical questions
- `/api/v1/health`: System health check
- Pydantic validation for requests/responses

### 📚 Enhanced Citations
- Document name + section in inline citations
- Similarity scores and token counts
- Text previews for verification
- Organ type classification

---

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Python 3.11+ with virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install uv package manager (optional but faster)
pip install uv

# Install Ollama (Windows)
winget install Ollama.Ollama
ollama pull phi3:mini
```

### 2. Install Dependencies
```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or standard pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 3. Build Knowledge Base
```bash
python scripts/build_kb.py build
```

Expected output:
```
✅ Knowledge base built successfully!
   Documents: 24
   Chunks: 194
   Total tokens: 37,018
   Time: 6.8s
```

### 4. Start API Server
```bash
python start_api.py
```

Server starts at: http://localhost:8000

### 5. View Documentation
Open browser to: http://localhost:8000/docs

### 6. Test API
```bash
python test_api.py
```

---

## 💻 Usage Examples

### Python Client
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "query": "What are signs of acute kidney rejection?",
        "top_k": 5,
        "max_tokens": 512
    }
)

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Confidence: {data['confidence']} ({data['confidence_score']})")

for source in data['sources']:
    print(f"\nSource: {source['document']}")
    print(f"Section: {source['section']}")
    print(f"Relevance: {source['similarity_score']}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are immunosuppressive drugs?",
    "top_k": 5
  }'
```

### Command Line (Alternative)
```bash
python scripts/simple_rag.py "What are signs of kidney rejection?"
```

---

## 📊 Example Response

```json
{
  "query": "What are signs of acute kidney rejection?",
  "answer": "- Increase in serum creatinine (>0.3 mg/dL above baseline)...",
  "confidence": "High",
  "confidence_score": 0.708,
  "sources": [
    {
      "document": "DOCUMENT 2: ACUTE REJECTION - MECHANISMS, DIAGNOSIS, AND TREATMENT",
      "section": "4.1 Clinical Features",
      "organ_type": "foundational",
      "similarity_score": 0.757,
      "token_count": 215
    }
  ],
  "retrieval_time": 0.232,
  "generation_time": 33.613,
  "total_time": 33.845,
  "chunks_used": 5,
  "total_tokens": 1076,
  "model": "phi3:mini"
}
```

---

## 🏗️ Architecture

### System Overview
```
┌─────────────────────────────────────┐
│         Client Layer                │
│  (Browser, Python, JavaScript)      │
└────────────┬────────────────────────┘
             │ HTTP POST /api/v1/query
             ▼
┌─────────────────────────────────────┐
│         FastAPI Layer               │
│  • Input validation (5-500 chars)   │
│  • Pydantic schemas                 │
│  • CORS middleware                  │
│  • Error handling                   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│      RAG Pipeline (pipeline.py)     │
│  1. Retrieve (top-k=5)              │
│  2. Compute confidence              │
│     • 60% avg similarity            │
│     • 30% top similarity            │
│     • 10% (1-variance)              │
│  3. Build prompt with context       │
│  4. Generate (Ollama, temp=0.05)    │
│  5. Format with citations           │
│  6. Log to queries.jsonl            │
└────────────┬────────────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌───────────┐  ┌───────────┐
│ ChromaDB  │  │  Ollama   │
│ 194 chunks│  │ phi3:mini │
│ mpnet emb │  │ GPU accel │
└───────────┘  └───────────┘
```

### Technologies
- **Web Framework:** FastAPI with Uvicorn
- **Vector DB:** ChromaDB with persistent storage
- **Embeddings:** sentence-transformers/all-mpnet-base-v2
- **LLM:** Ollama (phi3:mini) - GPU accelerated
- **NLP:** spaCy for text processing
- **Validation:** Pydantic for request/response schemas

### Confidence Scoring Algorithm
```python
confidence = 0.6×avg_similarity + 0.3×top_similarity + 0.1×(1-variance)
```

**Categories:**
- **High** (>0.70 with top >0.75): Strong evidence, reliable answer
- **Medium** (0.55-0.70): Moderate evidence, some gaps
- **Low** (<0.55): Weak evidence, verification needed

---

## 📡 API Reference

### Endpoints

#### Health Check
```bash
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "chroma_connected": true,
  "model_available": true,
  "chunks_indexed": 194
}
```

#### Query RAG System
```bash
POST /api/v1/query
Content-Type: application/json

{
  "query": "What are signs of acute rejection?",
  "top_k": 5,
  "max_tokens": 512
}
```

### Request Validation
- **Query length:** 5-500 characters
- **Prohibited terms:** Medical advice keywords (diagnose, dose, prescribe, etc.)
- **top_k range:** 1-10 (default: 5)
- **max_tokens:** 100-2000 (default: 512)

### Interactive Documentation
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 📊 Benchmarks

### Model Comparison Results

**Research Question:** Which compact LLM (≤4 GB VRAM) best balances medical accuracy and speed?

**Winner:** Gemma 3 (1B) - optimal for real-time clinical use

| Model | Parameters | Faithfulness | Avg Time | Tokens/sec |
|-------|------------|--------------|----------|------------|
| **Gemma 3** | 1B | **0.306** | **4.68s** | **109.6** |
| Gemma 2 | 2B | 0.301 | 9.06s | 56.6 |
| LLaMA 3.2 | 3B | **0.334** | 17.74s | 28.9 |
| Phi-3 Mini | 3.8B | 0.290 | 13.91s | 36.8 |

**Key Findings:**
- **Speed Champion:** Gemma 3 (3.8× faster than LLaMA 3.2)
- **Accuracy Leader:** LLaMA 3.2 (8.4% higher than Gemma 3)
- **Best Trade-off:** Gemma 3 (sub-5s responses with competitive accuracy)

**Hardware:** RTX 3050 (4GB VRAM), 16GB RAM  
**Test Date:** December 26, 2025  
**Test Queries:** 5 complex clinical reasoning questions

### Production Performance
- **Retrieval Time:** 0.2-0.5s
- **Generation Time:** 28-34s (phi3:mini)
- **Total Response:** 28-35s per query
- **Confidence Scoring:** Real-time calculation

---

## 📂 Project Structure

```
transplant_rag/
├── app/                    # FastAPI application
│   ├── main.py            # Application entry point
│   ├── api.py             # Route handlers
│   ├── schemas.py         # Pydantic models
│   ├── pipeline.py        # RAG pipeline + confidence
│   ├── security.py        # Authentication & security
│   └── middleware.py      # CORS & logging middleware
│
├── scripts/               # Core scripts
│   ├── build_kb.py        # Knowledge base builder
│   ├── retrieval.py       # Retrieval logic
│   ├── simple_rag.py      # CLI interface
│   ├── benchmark_rag.py   # Performance benchmarking
│   └── evaluate_rag.py    # Model evaluation
│
├── data/
│   ├── chroma/            # Vector database (generated)
│   ├── raw_docs/          # Source documents (24 files)
│   ├── chunks/            # Chunked text
│   └── metadata/          # Build manifests
│
├── logs/                  # Query logs (generated)
│   └── queries.jsonl
│
├── start_api.py           # Quick start script
├── start_frontend.py      # Streamlit UI (optional)
├── test_api.py            # API test suite
├── rag_config.toml        # System configuration
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker deployment
└── Dockerfile             # Container image
```

---

## 🛠️ Configuration

Edit `rag_config.toml` to adjust system behavior:

```toml
[embeddings]
model_name = "sentence-transformers/all-mpnet-base-v2"
device = "cuda"  # or "cpu"
batch_size = 16

[chroma]
anonymized_telemetry = false
collection_name = "medical_transplant_kb"
persist_directory = "data/chroma"

[chunking]
target_tokens = 400
overlap_tokens = 50
min_tokens = 100
max_tokens = 600

[retrieval]
top_k = 5
similarity_threshold = 0.3

[generation]
model = "phi3:mini"
temperature = 0.05
max_tokens = 512
```

---

## 🚀 Deployment

### Development Mode
```bash
# Start with auto-reload
python start_api.py

# Or directly with uvicorn
uvicorn app.main:app --reload
```

### Production Mode
```bash
# Multiple workers for high traffic
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Build image
docker build -t transplant-rag:latest .

# Run container
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  -e SECRET_KEY="your-secret-key" \
  transplant-rag:latest

# Or use docker-compose
docker-compose up -d
```

### Cloud Deployment Options

#### AWS EC2
- **Instance:** g4dn.xlarge (GPU) or t3.large (CPU)
- **AMI:** Deep Learning AMI (Ubuntu)
- **Storage:** 50GB EBS
- **Monthly Cost:** ~$60-200

#### Azure VM
- **VM Size:** NC6s v3 (GPU) or Standard_D2s_v3 (CPU)
- **Image:** Data Science VM - Ubuntu
- **Storage:** 50GB Premium SSD

#### Google Cloud
- **Machine Type:** n1-standard-2 with T4 GPU
- **Image:** Deep Learning VM
- **Boot disk:** 50GB SSD

---

## 🔒 Safety & Compliance

### Medical Safety Features
- ✅ Input validation (length, content)
- ✅ Prohibited terms blocking ("diagnose me", "what dose", etc.)
- ✅ Clear disclaimers in responses
- ✅ Confidence scoring for reliability assessment
- ✅ Source attribution for verification

### Audit & Monitoring
- ✅ Query logging with timestamps (`logs/queries.jsonl`)
- ✅ Confidence tracking per query
- ✅ Performance metrics (retrieval/generation time)
- ✅ Error logging with stack traces

### Production Checklist
- [x] Input validation
- [x] Query logging
- [x] Confidence scoring
- [x] Error handling
- [x] CORS configuration
- [ ] Authentication (add SECRET_KEY for production)
- [ ] Rate limiting (add for production)
- [ ] HTTPS/SSL (add for deployment)
- [ ] Backup strategy
- [ ] Monitoring dashboard

---

## 🧪 Testing

### Run Full Test Suite
```bash
python test_api.py
```

Expected output:
```
✓ Health check passed
✓ High confidence query: 0.708
✓ Low confidence query: 0.543 (properly flagged)
✓ Input validation working
✓ Query logging functional
```

### Manual API Testing
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Sample query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are immunosuppressive drugs?"}'
```

### Analyze Query Logs
```powershell
# View recent queries
Get-Content logs/queries.jsonl -Tail 10

# Count queries by confidence
Get-Content logs/queries.jsonl | ConvertFrom-Json | Group-Object confidence
```

---

## 🆘 Troubleshooting

### API won't start
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process using port
taskkill /PID <process_id> /F

# Try different port
uvicorn app.main:app --port 8001
```

### Ollama connection errors
```bash
# Check Ollama is running
ollama list

# Check model is available
ollama show phi3:mini

# Restart Ollama service (Windows)
# Close and reopen Ollama Desktop app
```

### ChromaDB errors
```bash
# Rebuild knowledge base
python scripts/build_kb.py build

# Check database exists
Test-Path data/chroma/chroma.sqlite3

# Clear and rebuild (if corrupted)
Remove-Item data/chroma -Recurse -Force
python scripts/build_kb.py build
```

### GPU not detected
```python
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Force CPU mode in rag_config.toml
# [embeddings]
# device = "cpu"
```

### Low confidence answers
1. Check query logs: `Get-Content logs/queries.jsonl -Tail 10`
2. Verify knowledge base: `python scripts/build_kb.py build`
3. Add more relevant documents to `data/raw_docs/`
4. Adjust retrieval parameters in `rag_config.toml`
5. Test with different models: `ollama pull llama3.2`

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🏆 Credits & Acknowledgments

**Technologies:**
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM inference
- [sentence-transformers](https://www.sbert.net/) - Embeddings
- [spaCy](https://spacy.io/) - NLP processing
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation

**Medical Content:** Clinical transplant medicine guidelines and protocols

---

## 📞 Support & Contact

For issues, questions, or contributions:
1. Check this README and troubleshooting section
2. Review API documentation at `/docs` endpoint
3. Examine logs: `logs/queries.jsonl`
4. Test system health: `python test_api.py`

---

**Built with ❤️ for safer, more informed clinical decision-making**

**Version:** 1.0.0  
**Status:** ✅ Production Ready  
**Last Updated:** December 27, 2025
