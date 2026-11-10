
import json
from pathlib import Path

import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(
        self,
        embedding_dimension: int = 384,
        storage_dir: str = "data/vector_store"
    ):
        if embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")

        self.embedding_dimension = embedding_dimension
        self.storage_dir = Path(storage_dir)

        self.index_path = self.storage_dir / "faiss.index"
        self.metadata_path = self.storage_dir / "documents.json"

        self.index = faiss.IndexFlatIP(embedding_dimension)
        self.documents = []

        self.load()

    def add_documents(self, documents: list[dict]) -> None:

        if not documents:
            return

        embeddings = []

        for document in documents:
            if "embedding" not in document:
                raise ValueError("Document is missing its embedding")

            embedding = np.asarray(
                document["embedding"],
                dtype=np.float32
            )

            if embedding.shape != (self.embedding_dimension,):
                raise ValueError("Invalid embedding dimension")

            if not np.all(np.isfinite(embedding)):
                raise ValueError("Embedding contains invalid numbers")

            norm = np.linalg.norm(embedding)

            if norm == 0:
                raise ValueError("Embedding cannot be a zero vector")

            embedding = embedding / norm
            embeddings.append(embedding)

        vectors = np.vstack(embeddings).astype(np.float32)

        self.index.add(vectors)
        self.documents.extend(documents)

        self.save()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[dict]:

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_vector.shape != (self.embedding_dimension,):
            raise ValueError("Invalid query embedding dimension")

        if not np.all(np.isfinite(query_vector)):
            raise ValueError("Query embedding contains invalid numbers")

        norm = np.linalg.norm(query_vector)

        if norm == 0:
            raise ValueError("Query embedding cannot be a zero vector")

        query_vector = query_vector / norm
        query_vector = np.expand_dims(query_vector, axis=0)

        actual_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            query_vector,
            actual_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            document = self.documents[index]

            results.append({
                **document,
                "similarity_score": float(score)
            })

        return results

    def save(self) -> None:

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(self.index_path)
        )

        metadata = [
            {
                key: value
                for key, value in document.items()
                if key != "embedding"
            }
            for document in self.documents
        ]

        with self.metadata_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(metadata, file, ensure_ascii=False)

    def load(self) -> None:

        if not self.index_path.exists() and not self.metadata_path.exists():
            return

        if not self.index_path.exists() or not self.metadata_path.exists():
            raise RuntimeError(
                "FAISS index or metadata file is missing"
            )

        loaded_index = faiss.read_index(str(self.index_path))

        with self.metadata_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            loaded_documents = json.load(file)

        if loaded_index.d != self.embedding_dimension:
            raise ValueError("Stored index dimension does not match")

        if loaded_index.ntotal != len(loaded_documents):
            raise ValueError(
                "FAISS index and document metadata are inconsistent"
            )

        self.index = loaded_index
        self.documents = loaded_documents

    def get_total_documents(self) -> int:
        return self.index.ntotal


vector_store = FAISSVectorStore()
