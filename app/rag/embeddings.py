# TODO: Generate embeddings for document chunks.

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingService:

    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"

        self.model = SentenceTransformer(
            self.model_name
        )

        self.embedding_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

    def generate_embedding(self, text: str) -> list[float]:

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def generate_chunk_embeddings(
        self,
        chunks: list[dict]
    ) -> list[dict]:

        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        results = []

        for chunk, embedding in zip(chunks, embeddings):

            results.append({
                **chunk,
                "embedding": embedding.tolist()
            })

        return results

    def cosine_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float]
    ) -> float:

        vector1 = np.array(embedding1)
        vector2 = np.array(embedding2)

        if vector1.shape != vector2.shape:
            raise ValueError("Embedding dimensions must match")

        if vector1.ndim != 1 or vector1.size == 0:
            raise ValueError("Embeddings must be non-empty vectors")

        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Embeddings cannot be zero vectors")

        return float(
            np.dot(vector1, vector2) / (norm1 * norm2)
        )


embedding_service = EmbeddingService()
