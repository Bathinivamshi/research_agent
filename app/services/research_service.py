
import asyncio
import hashlib
from urllib.parse import urlsplit, urlunsplit

from app.tools.web_search import web_search_service
from app.tools.web_scraper import web_scraper_service


class ResearchService:

    def __init__(self):
        self.max_concurrent_scrapes = 5

    @staticmethod
    def generate_source_id(url: str) -> str:
        """Generate a deterministic ID from a normalized webpage URL."""

        if not url or not url.strip():
            raise ValueError("URL cannot be empty")

        parsed = urlsplit(url.strip())

        normalized_url = urlunsplit((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.query,
            ""
        ))

        url_hash = hashlib.sha256(
            normalized_url.encode("utf-8")
        ).hexdigest()

        return f"web_{url_hash[:16]}"

    async def scrape_source(self, search_result: dict) -> dict:

        url = search_result.get("url", "")

        try:
            scraped_data = await web_scraper_service.scrape(url)

            final_url = scraped_data["url"]

            return {
                "source_id": self.generate_source_id(final_url),
                "title": search_result.get("title", ""),
                "url": final_url,
                "content": scraped_data["content"],
                "content_length": scraped_data["content_length"],
                "score": search_result.get("score", 0.0),
                "published_date": search_result.get("published_date"),
                "status": "success"
            }

        except Exception as error:
            return {
                "source_id": (
                    self.generate_source_id(url) if url else None
                ),
                "title": search_result.get("title", ""),
                "url": url,
                "content": "",
                "content_length": 0,
                "score": search_result.get("score", 0.0),
                "published_date": search_result.get("published_date"),
                "status": "failed",
                "error": str(error)
            }

    async def research(
        self,
        query: str,
        max_results: int = 5
    ) -> dict:

        if not query or not query.strip():
            raise ValueError("Research query cannot be empty")

        if not 1 <= max_results <= 10:
            raise ValueError("max_results must be between 1 and 10")

        search_results = await web_search_service.search(
            query=query,
            max_results=max_results
        )

        semaphore = asyncio.Semaphore(
            self.max_concurrent_scrapes
        )

        async def scrape_with_limit(result):
            async with semaphore:
                return await self.scrape_source(result)

        tasks = [
            scrape_with_limit(result)
            for result in search_results
        ]

        research_documents = await asyncio.gather(*tasks)

        successful_documents = [
            document for document in research_documents
            if document["status"] == "success"
        ]

        failed_documents = [
            document for document in research_documents
            if document["status"] == "failed"
        ]

        return {
            "query": query,
            "total_search_results": len(search_results),
            "successful_scrapes": len(successful_documents),
            "failed_scrapes": len(failed_documents),
            "documents": research_documents
        }


research_service = ResearchService()
