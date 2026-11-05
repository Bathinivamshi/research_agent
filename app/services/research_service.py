
import asyncio

from app.tools.web_search import web_search_service
from app.tools.web_scraper import web_scraper_service


class ResearchService:

    def __init__(self):

        self.max_concurrent_scrapes = 5

    async def scrape_source(self, search_result: dict) -> dict:

        url = search_result.get("url", "")

        try:

            scraped_data = await web_scraper_service.scrape(url)

            return {
                "title": search_result.get("title", ""),
                "url": scraped_data["url"],
                "content": scraped_data["content"],
                "content_length": scraped_data["content_length"],
                "score": search_result.get("score", 0.0),
                "published_date": search_result.get("published_date"),
                "status": "success"
            }

        except Exception as error:

            return {
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

        # Step 1: Search the web
        search_results = await web_search_service.search(
            query=query,
            max_results=max_results
        )

        # Step 2: Limit concurrent webpage requests
        semaphore = asyncio.Semaphore(
            self.max_concurrent_scrapes
        )

        async def scrape_with_limit(result):

            async with semaphore:
                return await self.scrape_source(result)

        # Step 3: Scrape multiple websites concurrently
        tasks = [
            scrape_with_limit(result)
            for result in search_results
        ]

        research_documents = await asyncio.gather(*tasks)

        # Step 4: Separate successful and failed sources
        successful_documents = [
            document
            for document in research_documents
            if document["status"] == "success"
        ]

        failed_documents = [
            document
            for document in research_documents
            if document["status"] == "failed"
        ]

        # Step 5: Return research data
        return {
            "query": query,
            "total_search_results": len(search_results),
            "successful_scrapes": len(successful_documents),
            "failed_scrapes": len(failed_documents),
            "documents": research_documents
        }


research_service = ResearchService()
