
from uuid import uuid4

from app.tools.pdf_reader import pdf_reader_service
from app.rag.chunking import document_chunker
from app.rag.embeddings import embedding_service
from app.rag.vector_store import FAISSVectorStore


class DocumentIngestionService:

    def __init__(self):
        self.vector_store = FAISSVectorStore(
            embedding_dimension=384
        )

    def store_chunks(
        self,
        chunks: list[dict],
        document_id: str,
        document_name: str,
        source_type: str,
        source_url: str = ""
    ) -> int:

        if not chunks:
            return 0

        for chunk in chunks:
            chunk["document_id"] = document_id
            chunk["document_name"] = document_name
            chunk["source_type"] = source_type
            chunk["source_url"] = source_url

        embedded_chunks = (
            embedding_service.generate_chunk_embeddings(chunks)
        )

        self.vector_store.add_documents(embedded_chunks)

        return len(embedded_chunks)

    def ingest_pdf(self, pdf_bytes: bytes) -> dict:

        document = pdf_reader_service.extract_text(pdf_bytes)

        chunks = document_chunker.chunk_document(document)

        if not chunks:
            raise ValueError(
                "No readable text found in the PDF"
            )

        document_id = str(uuid4())

        total_chunks = self.store_chunks(
            chunks=chunks,
            document_id=document_id,
            document_name=document["filename"],
            source_type="pdf"
        )

        return {
            "document_id": document_id,
            "filename": document["filename"],
            "total_pages": document["total_pages"],
            "total_chunks": total_chunks,
            "total_vectors": self.vector_store.get_total_documents(),
            "message": "PDF successfully ingested"
        }

    def ingest_web_source(self, source: dict) -> dict:

        title = source.get("title", "").strip()
        url = source.get("url", "").strip()
        content = source.get("content", "").strip()

        if not title:
            raise ValueError("Web source title cannot be empty")

        if not url:
            raise ValueError("Web source URL cannot be empty")

        if not content:
            raise ValueError("Web source content cannot be empty")

        document_id = str(uuid4())

        web_document = {
            "filename": title,
            "pages": [
                {
                    "page_number": 1,
                    "content": content
                }
            ]
        }

        chunks = document_chunker.chunk_document(web_document)

        total_chunks = self.store_chunks(
            chunks=chunks,
            document_id=document_id,
            document_name=title,
            source_type="web",
            source_url=url
        )

        return {
            "document_id": document_id,
            "title": title,
            "url": url,
            "total_chunks": total_chunks,
            "message": "Web source successfully ingested"
        }

    def search_documents(
        self,
        query: str,
        top_k: int = 5
    ) -> list:

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        query_embedding = (
            embedding_service.generate_embedding(query)
        )

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )


document_ingestion_service = DocumentIngestionService()
