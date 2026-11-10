
import pytest

from app.rag.citations import citation_service


def test_format_citation():

    document = {
        "document_id": "doc-123",
        "document_name": "research.pdf",
        "page_number": 4,
        "chunk_index": 0
    }

    result = citation_service.format_citation(document)

    assert result["reference"] == "research.pdf, Page 4"
    assert result["citation_id"] == "doc-123-p4-c0"


def test_format_search_results():

    results = [
        {
            "text": "Machine learning finds patterns in data.",
            "document_id": "doc-123",
            "document_name": "research.pdf",
            "page_number": 4,
            "chunk_index": 0,
            "similarity_score": 0.85
        }
    ]

    formatted = citation_service.format_search_results(results)

    assert len(formatted) == 1
    assert formatted[0]["citation"]["page_number"] == 4
    assert formatted[0]["similarity_score"] == 0.85


def test_missing_page_number_raises_error():

    document = {
        "document_id": "doc-123",
        "document_name": "research.pdf",
        "chunk_index": 0
    }

    with pytest.raises(ValueError):
        citation_service.format_citation(document)


def test_missing_chunk_index_raises_error():

    document = {
        "document_id": "doc-123",
        "document_name": "research.pdf",
        "page_number": 4
    }

    with pytest.raises(ValueError):
        citation_service.format_citation(document)
