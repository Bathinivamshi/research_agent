
import io

import pymupdf
import pytest

from app.rag.ingestion import DocumentIngestionService


def create_sample_pdf():
    """Create a small PDF in memory for testing."""

    pdf = pymupdf.open()

    page1 = pdf.new_page()
    page1.insert_text(
        (72, 72),
        (
            "Machine learning is a branch of artificial intelligence. "
            "It enables computers to learn patterns from data. "
        ) * 15
    )

    page2 = pdf.new_page()
    page2.insert_text(
        (72, 72),
        (
            "Deep learning uses neural networks with multiple layers. "
            "It is widely used in computer vision and NLP. "
        ) * 15
    )

    pdf_bytes = pdf.tobytes()
    pdf.close()

    return pdf_bytes


@pytest.fixture
def ingestion_service():
    return DocumentIngestionService()


def test_ingest_pdf(ingestion_service):
    pdf_bytes = create_sample_pdf()

    result = ingestion_service.ingest_pdf(pdf_bytes)

    assert result["total_pages"] == 2
    assert result["total_chunks"] > 0
    assert result["total_vectors"] == result["total_chunks"]
    assert result["message"] == "PDF successfully ingested"


def test_search_ingested_document(ingestion_service):
    pdf_bytes = create_sample_pdf()

    ingestion_service.ingest_pdf(pdf_bytes)

    results = ingestion_service.search_documents(
        query="How do computers learn from data?",
        top_k=3
    )

    assert len(results) > 0
    assert all("text" in item for item in results)
    assert all("page_number" in item for item in results)
    assert all("similarity_score" in item for item in results)


def test_empty_query_raises_error(ingestion_service):
    with pytest.raises(ValueError):
        ingestion_service.search_documents("")


def test_invalid_pdf_raises_error(ingestion_service):
    with pytest.raises(ValueError):
        ingestion_service.ingest_pdf(b"This is not a PDF")
