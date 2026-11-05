
from fastapi import FastAPI

from app.config import settings


app = FastAPI(
    title=settings.app_name,
    description=(
        "An AI-powered research agent that searches "
        "the web and documents to generate reports "
        "with verified citations."
    ),
    version="0.1.0"
)


@app.get("/")
async def root():

    return {
        "message": "Welcome to Research Agent with Cited Reports",
        "status": "running",
        "environment": settings.app_env
    }


@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "application": settings.app_name
    }
