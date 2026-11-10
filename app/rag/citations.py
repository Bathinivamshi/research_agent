
class CitationService:

    def format_citation(self, document: dict) -> dict:

        document_name = document.get(
            "document_name",
            "Unknown document"
        )

        document_id = document.get(
            "document_id",
            "unknown"
        )

        page_number = document.get("page_number")

        if page_number is None:
            raise ValueError("Citation requires a page number")

        chunk_index = document.get("chunk_index")

        if chunk_index is None:
            raise ValueError("Citation requires a chunk index")

        return {
            "citation_id": (
                f"{document_id}-p{page_number}-c{chunk_index}"
            ),
            "document_id": document_id,
            "document_name": document_name,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "reference": f"{document_name}, Page {page_number}"
        }

    def format_search_results(
        self,
        search_results: list[dict]
    ) -> list[dict]:

        formatted_results = []

        for result in search_results:

            citation = self.format_citation(result)

            formatted_results.append({
                "text": result.get("text", ""),
                "similarity_score": result.get(
                    "similarity_score",
                    0.0
                ),
                "citation": citation
            })

        return formatted_results


citation_service = CitationService()
