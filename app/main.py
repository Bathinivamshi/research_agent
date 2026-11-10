
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.config import settings
from app.services.llm_service import llm_service
from app.services.research_service import research_service
from app.services.web_research_ingestion import (
    web_research_ingestion_service
)

from app.tools.web_search import web_search_service
from app.tools.pdf_reader import pdf_reader_service

from app.rag.ingestion import document_ingestion_service
from app.rag.query_engine import rag_query_engine

from app.graph.workflow import research_workflow


# -------------------------
# FASTAPI APPLICATION
# -------------------------

app = FastAPI(
    title=settings.app_name,
    description="AI-powered multi-agent research system with cited reports",
    version="0.5.0"
)


# -------------------------
# REQUEST AND RESPONSE MODELS
# -------------------------

class LLMRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=10000)


class LLMResponse(BaseModel):
    response: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    max_results: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[dict]


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    max_results: int = Field(default=5, ge=1, le=10)


class ResearchResponse(BaseModel):
    query: str
    total_search_results: int
    successful_scrapes: int
    failed_scrapes: int
    documents: list[dict]


class RAGQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RAGSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class WebResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    max_results: int = Field(default=5, ge=1, le=10)


class AgentResearchRequest(BaseModel):
    research_question: str = Field(
        ...,
        min_length=5,
        max_length=1000
    )


# -------------------------
# ROOT ENDPOINT
# -------------------------

@app.get("/")
async def root():
    return {
        "message": "Welcome to Research Agent with Cited Reports",
        "status": "running",
        "environment": settings.app_env,
        "version": app.version
    }


# -------------------------
# HEALTH CHECK
# -------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "application": settings.app_name
    }


# -------------------------
# GEMINI LLM ENDPOINT
# -------------------------

@app.post(
    "/api/llm/generate",
    response_model=LLMResponse
)
async def generate_llm_response(request: LLMRequest):
    try:
        result = await llm_service.generate_response(
            request.prompt
        )

        return LLMResponse(response=result)

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Gemini request failed. Check terminal logs."
        )


# -------------------------
# TAVILY SEARCH ENDPOINT
# -------------------------

@app.post(
    "/api/search",
    response_model=SearchResponse
)
async def search_web(request: SearchRequest):
    try:
        results = await web_search_service.search(
            query=request.query,
            max_results=request.max_results
        )

        return SearchResponse(
            query=request.query,
            total_results=len(results),
            results=results
        )

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Web search failed. Check terminal logs."
        )


# -------------------------
# COMPLETE WEB RESEARCH ENDPOINT
# -------------------------

@app.post(
    "/api/research",
    response_model=ResearchResponse
)
async def conduct_research(request: ResearchRequest):
    try:
        result = await research_service.research(
            query=request.query,
            max_results=request.max_results
        )

        return ResearchResponse(**result)

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Research process failed. Check terminal logs."
        )


# -------------------------
# PDF DOCUMENT UPLOAD
# -------------------------

@app.post("/api/documents/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if (
            not file.filename
            or not file.filename.lower().endswith(".pdf")
        ):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )

        pdf_bytes = await file.read()

        result = pdf_reader_service.extract_text(pdf_bytes)
        result["filename"] = file.filename

        return {
            "message": "PDF processed successfully",
            "document": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="PDF processing failed"
        )

    finally:
        await file.close()


# -------------------------
# RAG DOCUMENT INGESTION
# -------------------------

@app.post("/api/rag/ingest")
async def ingest_document(file: UploadFile = File(...)):
    try:
        if (
            not file.filename
            or not file.filename.lower().endswith(".pdf")
        ):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )

        pdf_bytes = await file.read()

        result = document_ingestion_service.ingest_pdf(pdf_bytes)

        return {
            "message": "Document ingested successfully",
            "result": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed"
        )

    finally:
        await file.close()


# -------------------------
# RAG DOCUMENT SEARCH
# -------------------------

@app.post("/api/rag/search")
async def search_documents(request: RAGSearchRequest):
    try:
        results = document_ingestion_service.search_documents(
            query=request.query,
            top_k=request.top_k
        )

        return {
            "query": request.query,
            "total_results": len(results),
            "results": results
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Document search failed"
        )


# -------------------------
# RAG QUESTION ANSWERING
# -------------------------

@app.post("/api/rag/ask")
async def ask_question(request: RAGQuestionRequest):
    try:
        result = await rag_query_engine.answer_question(
            question=request.question,
            top_k=request.top_k
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Question answering failed"
        )


# -------------------------
# WEB RESEARCH AND INGESTION
# -------------------------

@app.post("/api/rag/research")
async def research_web(request: WebResearchRequest):
    try:
        result = await web_research_ingestion_service.research_and_ingest(
            query=request.query,
            max_results=request.max_results
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Web research ingestion failed"
        )


# -------------------------
# COMPLETE AGENTIC RESEARCH WORKFLOW
# -------------------------

@app.post("/api/agent/research")
async def run_agentic_research(request: AgentResearchRequest):
    try:
        result = await research_workflow.run_research(
            research_question=request.research_question
        )

        return {
            "message": "Agentic research workflow completed",
            "status": result.get("status"),
            "research_question": result.get("research_question"),
            "research_plan": result.get("plan"),
            "total_sources_collected": result.get(
                "total_sources_collected", 0
            ),
            "verified_evidence": result.get(
                "verified_evidence", []
            ),
            "rejected_evidence": result.get(
                "rejected_evidence", []
            ),
            "verification_passed": result.get(
                "verification_passed", False
            ),
            "report": result.get("report", {})
        }

    except Exception as error:
        import logging

        logging.exception("Agentic research workflow failed")

        raise HTTPException(
            status_code=500,
            detail=f"Agentic research workflow failed: {str(error)}"
        )