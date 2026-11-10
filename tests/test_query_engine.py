
import pytest

from app.rag.ingestion import DocumentIngestionService
from app.rag.query_engine import RAGQueryEngine


class MockAnswerGenerator:

    async def generate_answer(self, question, evidence_chunks):
        return (
            "Machine learning allows computers to learn from data. "
            "[Source: research.pdf, Page: 1]"
        )


class MockEvidenceVerifier:

    def verify_claim(self, claim, evidence_chunks):
        return {
            "claim": claim,
            "verified": True,
            "reason": "Supporting evidence found",
            "evidence": evidence_chunks
        }


@pytest.fixture
def query_engine():

    ingestion = DocumentIngestionService()

    return RAGQueryEngine(
        ingestion_service=ingestion,
        answer_generator_instance=MockAnswerGenerator(),
        evidence_verifier_instance=MockEvidenceVerifier()
    )


@pytest.mark.asyncio
async def test_answer_question_returns_generated_answer(query_engine):

    query_engine.ingestion_service.search_documents = lambda query, top_k: [
        {
            "text": "Machine learning allows computers to learn from data.",
            "document_id": "doc-123",
            "document_name": "research.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "similarity_score": 0.85
        }
    ]

    result = await query_engine.answer_question(
        "What is machine learning?",
        top_k=3
    )

    assert "Machine learning" in result["answer"]
    assert result["verified"] is True
    assert len(result["sources"]) == 1
    assert result["sources"][0]["page_number"] == 1


@pytest.mark.asyncio
async def test_empty_question_raises_error(query_engine):

    with pytest.raises(ValueError):
        await query_engine.answer_question("")


@pytest.mark.asyncio
async def test_no_documents_returns_empty_sources(query_engine):

    query_engine.ingestion_service.search_documents = lambda query, top_k: []

    result = await query_engine.answer_question(
        "What is machine learning?"
    )

    assert result["sources"] == []
    assert result["verified"] is False
