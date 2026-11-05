
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.services.llm_service import llm_service
from app.services.research_service import research_service
from app.tools.web_search import web_search_service


app = FastAPI(
    title=settings.app_name,
    description="AI-powered research agent with cited reports",
    version="0.4.0"
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


# -------------------------
# ROOT ENDPOINT
# -------------------------

@app.get("/")
async def root():

    return {
        "message": "Welcome to Research Agent with Cited Reports",
        "status": "running",
        "environment": settings.app_env
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
# COMPLETE RESEARCH ENDPOINT
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
