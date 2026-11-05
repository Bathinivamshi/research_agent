
from tavily import AsyncTavilyClient

from app.config import settings


class WebSearchService:

    def __init__(self):

        if not settings.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not configured")

        self.client = AsyncTavilyClient(
            api_key=settings.tavily_api_key
        )

    async def search(
        self,
        query: str,
        max_results: int = 5
    ) -> list:

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")

        response = await self.client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False
        )

        results = []

        for item in response.get("results", []):

            results.append({

                "title": item.get("title", ""),

                "url": item.get("url", ""),

                "content": item.get("content", ""),

                "score": item.get("score", 0.0),

                "published_date": item.get(
                    "published_date"
                )

            })

        return results


web_search_service = WebSearchService()
