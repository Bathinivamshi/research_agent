
import pytest

from app.rag.chunking import document_chunker


def create_sample_document():
    return {
        "filename": "sample.pdf",
        "total_pages": 2,
        "pages": [
            {
                "page_number": 1,
                "content": (
                    "Machine learning is a branch of artificial intelligence. "
                    "It enables computers to learn patterns from data. "
                ) * 30
            },
            {
                "page_number": 2,
                "content": (
                    "Deep learning uses neural networks with multiple layers. "
                    "It is widely used in computer vision and NLP. "
                ) * 30
            }
        ]
    }


def test_chunk_document_creates_chunks():
    document = create_sample_document()

    chunks = document_chunker.chunk_document(document)

    assert len(chunks) > 0


def test_chunks_preserve_page_numbers():
    document = create_sample_document()

    chunks = document_chunker.chunk_document(document)

    page_numbers = {chunk["page_number"] for chunk in chunks}

    assert page_numbers == {1, 2}


def test_chunks_do_not_exceed_chunk_size():
    document = create_sample_document()

    chunks = document_chunker.chunk_document(document)

    for chunk in chunks:
        assert chunk["character_count"] <= 1000


def test_chunks_contain_required_fields():
    document = create_sample_document()

    chunks = document_chunker.chunk_document(document)

    for chunk in chunks:
        assert "text" in chunk
        assert "page_number" in chunk
        assert "chunk_index" in chunk
        assert "character_count" in chunk
        assert chunk["text"].strip()


def test_empty_document_raises_error():
    document = {
        "filename": "empty.pdf",
        "pages": []
    }

    with pytest.raises(ValueError):
        document_chunker.chunk_document(document)
