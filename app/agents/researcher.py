# TODO: Coordinate web and document research tasks.

import asyncio

from app.services.research_service import research_service


class ResearcherAgent:

    def __init__(self, research_service_instance=None):
        self.research_service = (
            research_service_instance or research_service
        )

        self.max_concurrent_queries = 3

    async def execute_query(self, query: str) -> dict:

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        try:
            result = await self.research_service.research(
                query=query,
                max_results=5
            )

            return {
                "query": query,
                "total_search_results": result[
                    "total_search_results"
                ],
                "successful_scrapes": result[
                    "successful_scrapes"
                ],
                "failed_scrapes": result[
                    "failed_scrapes"
                ],
                "documents": result["documents"],
                "status": "success"
            }

        except Exception as error:
            return {
                "query": query,
                "total_search_results": 0,
                "successful_scrapes": 0,
                "failed_scrapes": 0,
                "documents": [],
                "status": "failed",
                "error": str(error)
            }

    async def execute_research_plan(
        self,
        search_queries: list[str]
    ) -> dict:

        if not search_queries:
            raise ValueError("Search queries cannot be empty")

        if not all(
            isinstance(query, str) and query.strip()
            for query in search_queries
        ):
            raise ValueError("Search queries must be non-empty strings")

        semaphore = asyncio.Semaphore(
            self.max_concurrent_queries
        )

        async def execute_with_limit(query):
            async with semaphore:
                return await self.execute_query(query)

        tasks = [
            execute_with_limit(query)
            for query in search_queries
        ]

        results = await asyncio.gather(*tasks)

        all_documents = []

        for result in results:
            if result["status"] == "success":
                all_documents.extend(result["documents"])

        successful_queries = [
            result for result in results
            if result["status"] == "success"
        ]

        failed_queries = [
            result for result in results
            if result["status"] == "failed"
        ]

        return {
            "total_queries": len(search_queries),
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "total_sources_collected": len(all_documents),
            "documents": all_documents,
            "query_results": results
        }


researcher_agent = ResearcherAgent()
