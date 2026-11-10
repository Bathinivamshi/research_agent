
from app.rag.embeddings import embedding_service


def test_embedding_generation():
    text = "Machine learning helps computers learn from data."

    embedding = embedding_service.generate_embedding(text)

    assert isinstance(embedding, list)
    assert len(embedding) == 384


def test_similar_sentences_have_high_similarity():
    sentence1 = "Machine learning helps computers learn from data."
    sentence2 = "Computers can learn patterns using machine learning."

    embedding1 = embedding_service.generate_embedding(sentence1)
    embedding2 = embedding_service.generate_embedding(sentence2)

    similarity = embedding_service.cosine_similarity(
        embedding1,
        embedding2
    )

    print("Similarity:", similarity)

    assert similarity > 0.5


def test_unrelated_sentences_have_lower_similarity():
    sentence1 = "Machine learning helps computers learn from data."
    sentence2 = "The weather is sunny today."

    embedding1 = embedding_service.generate_embedding(sentence1)
    embedding2 = embedding_service.generate_embedding(sentence2)

    similarity = embedding_service.cosine_similarity(
        embedding1,
        embedding2
    )

    print("Unrelated similarity:", similarity)

    assert similarity < 0.8


def test_chunk_embedding_generation():
    chunks = [
        {
            "text": "Machine learning is a branch of artificial intelligence.",
            "page_number": 1,
            "chunk_index": 0,
            "character_count": 55
        },
        {
            "text": "Deep learning uses neural networks.",
            "page_number": 2,
            "chunk_index": 0,
            "character_count": 36
        }
    ]

    results = embedding_service.generate_chunk_embeddings(chunks)

    assert len(results) == 2
    assert all(len(item["embedding"]) == 384 for item in results)
    assert all("page_number" in item for item in results)


def test_empty_text_raises_error():
    try:
        embedding_service.generate_embedding("")
        assert False, "Expected ValueError"
    except ValueError:
        pass
