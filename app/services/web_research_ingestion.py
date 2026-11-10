
from app.services.research_service import research_service
from app.rag.ingestion import document_ingestion_service


class WebResearchIngestionService:

    async def research_and_ingest(
        self,
        query: str,
        max_results: int = 5
    ) -> dict:

        if not query or not query.strip():
            raise ValueError("Research query cannot be empty")

        if not 1 <= max_results <= 10:
            raise ValueError("max_results must be between 1 and 10")

        # Step 1: Search the web and scrape webpages.
        research_results = await research_service.research(
            query=query,
            max_results=max_results
        )

        # Step 2: Keep only successfully scraped sources.
        successful_documents = [
            document
            for document in research_results["documents"]
            if document["status"] == "success"
        ]

        # Step 3: Ingest each webpage into the shared FAISS store.
        ingested_sources = []
        failed_sources = []

        for document in successful_documents:

            try:
                result = document_ingestion_service.ingest_web_source(
                    {
                        "title": document["title"],
                        "url": document["url"],
                        "content": document["content"]
                    }
                )

                ingested_sources.append(result)

            except Exception as error:
                failed_sources.append({
                    "title": document.get("title", ""),
                    "url": document.get("url", ""),
                    "error": str(error)
                })

        return {
            "query": query,
            "total_search_results": research_results[
                "total_search_results"
            ],
            "successful_scrapes": research_results[
                "successful_scrapes"
            ],
            "failed_scrapes": research_results[
                "failed_scrapes"
            ],
            "successfully_ingested": len(ingested_sources),
            "ingested_sources": ingested_sources,
            "failed_ingestions": failed_sources,
            "total_vectors": (
                document_ingestion_service
                .vector_store
                .get_total_documents()
            )
        }


web_research_ingestion_service = WebResearchIngestionService()
