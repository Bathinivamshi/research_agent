
import pytest

from app.rag.embeddings import embedding_service
from app.rag.vector_store import FAISSVectorStore


@pytest.fixture
def vector_store():
    return FAISSVectorStore(embedding_dimension=384)


def test_add_documents(vector_store):
    documents = [
        {
            "text": "Machine learning helps computers learn from data.",
            "page_number": 1,
            "embedding": embedding_service.generate_embedding(
                "Machine learning helps computers learn from data."
            )
        },
        {
            "text": "Deep learning uses neural networks.",
            "page_number": 2,
            "embedding": embedding_service.generate_embedding(
                "Deep learning uses neural networks."
            )
        }
    ]

    vector_store.add_documents(documents)

    assert vector_store.get_total_documents() == 2


def test_search_returns_relevant_document(vector_store):
    documents = [
        {
            "text": "Machine learning helps computers learn from data.",
            "page_number": 1,
            "embedding": embedding_service.generate_embedding(
                "Machine learning helps computers learn from data."
            )
        },
        {
            "text": "Deep learning uses neural networks.",
            "page_number": 2,
            "embedding": embedding_service.generate_embedding(
                "Deep learning uses neural networks."
            )
        },
        {
            "text": "The weather is sunny today.",
            "page_number": 3,
            "embedding": embedding_service.generate_embedding(
                "The weather is sunny today."
            )
        }
    ]

    vector_store.add_documents(documents)

    query_embedding = embedding_service.generate_embedding(
        "How do computers learn from data?"
    )

    results = vector_store.search(query_embedding, top_k=2)

    assert len(results) == 2
    assert results[0]["page_number"] == 1
    assert "similarity_score" in results[0]


def test_search_empty_store_returns_empty_list(vector_store):
    query_embedding = embedding_service.generate_embedding(
        "What is machine learning?"
    )

    results = vector_store.search(query_embedding)

    assert results == []


def test_top_k_cannot_be_zero(vector_store):
    query_embedding = embedding_service.generate_embedding(
        "What is machine learning?"
    )

    with pytest.raises(ValueError):
        vector_store.search(query_embedding, top_k=0)


def test_invalid_embedding_dimension(vector_store):
    document = {
        "text": "Sample text",
        "embedding": [0.1, 0.2]
    }

    with pytest.raises(ValueError):
        vector_store.add_documents([document])
