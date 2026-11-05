
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.services.llm_service import llm_service
import logging

logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.app_name,
    description="AI-powered research agent with cited reports",
    version="0.2.0"
)


# Request model
class LLMRequest(BaseModel):

    prompt: str = Field(
        ...,
        min_length=3,
        max_length=10000
    )


# Response model
class LLMResponse(BaseModel):

    response: str


# Root endpoint
@app.get("/")
async def root():

    return {
        "message": "Welcome to Research Agent with Cited Reports",
        "status": "running",
        "environment": settings.app_env
    }


# Health check
@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "application": settings.app_name
    }


# Gemini generation endpoint
@app.post("/api/llm/generate", response_model=LLMResponse)
async def generate_llm_response(request: LLMRequest):

    try:

        result = await llm_service.generate_response(
            request.prompt
        )

        return LLMResponse(response=result)

    except Exception as e:
        logger.exception("Gemini API request failed")

        raise HTTPException(
            status_code=502,
            detail=str(e)
        )