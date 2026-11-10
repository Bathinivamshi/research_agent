# TODO: Split source documents into retrieval-friendly chunks.

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero")

        if chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

    def chunk_document(self, document: dict) -> list:
        pages = document.get("pages", [])

        if not pages:
            raise ValueError("Document contains no pages")

        chunks = []

        for page in pages:
            page_number = page["page_number"]
            page_content = page["content"].strip()

            if not page_content:
                continue

            page_chunks = self.splitter.split_text(page_content)

            for chunk_index, chunk_text in enumerate(page_chunks):
                chunks.append({
                    "text": chunk_text,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "character_count": len(chunk_text)
                })

        return chunks


document_chunker = DocumentChunker()
