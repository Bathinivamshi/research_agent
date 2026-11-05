
# Research Agent with Cited Reports

An AI-powered research agent that takes a research question, searches the web and documents, extracts relevant evidence, verifies claims, and generates structured reports with citations.

## Features

- Automated research question decomposition
- Parallel web research
- PDF document processing
- Retrieval-Augmented Generation (RAG)
- Multi-agent orchestration using LangGraph
- Evidence extraction and verification
- Claim-level source citations
- Contradictory evidence detection
- Research report generation
- LLM evaluation and cost tracking

## Technology Stack

- Python
- FastAPI
- Gemini API
- LangChain
- LangGraph
- Tavily
- PyMuPDF
- Sentence Transformers
- FAISS
- PostgreSQL
- Redis
- Ragas
- DeepEval
- Docker

## Project Status

Currently under development.

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd research-agent
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file:

```bash
copy .env.example .env
```

Add your Gemini and Tavily API keys to `.env`.

## Run the Application

```bash
uvicorn app.main:app --reload
```

Open:

- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health

## Run Tests

```bash
pytest
```

## License

MIT
