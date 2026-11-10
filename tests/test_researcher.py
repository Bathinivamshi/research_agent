
import pytest

from app.agents.researcher import ResearcherAgent


class MockResearchService:

    async def research(self, query, max_results=5):

        if "failing" in query.lower():
            raise Exception("Simulated search failure")

        return {
            "total_search_results": 2,
            "successful_scrapes": 2,
            "failed_scrapes": 0,
            "documents": [
                {
                    "title": "AI Research Paper",
                    "url": "https://example.com/research",
                    "content": "Research content about artificial intelligence.",
                    "status": "success"
                },
                {
                    "title": "AI Industry Report",
                    "url": "https://example.com/report",
                    "content": "Industry report about AI applications.",
                    "status": "success"
                }
            ]
        }


@pytest.fixture
def researcher():
    return ResearcherAgent(
        research_service_instance=MockResearchService()
    )


@pytest.mark.asyncio
async def test_execute_single_query(researcher):

    result = await researcher.execute_query(
        "Artificial intelligence research"
    )

    assert result["status"] == "success"
    assert result["total_search_results"] == 2
    assert len(result["documents"]) == 2


@pytest.mark.asyncio
async def test_execute_research_plan(researcher):

    result = await researcher.execute_research_plan([
        "AI research",
        "Machine learning research",
        "Deep learning research"
    ])

    assert result["total_queries"] == 3
    assert result["successful_queries"] == 3
    assert result["total_sources_collected"] == 6


@pytest.mark.asyncio
async def test_failed_query_does_not_stop_other_queries(researcher):

    result = await researcher.execute_research_plan([
        "AI research",
        "Failing query"
    ])

    assert result["successful_queries"] == 1
    assert result["failed_queries"] == 1
    assert result["total_sources_collected"] == 2


@pytest.mark.asyncio
async def test_empty_query_raises_error(researcher):

    with pytest.raises(ValueError):
        await researcher.execute_query("")


@pytest.mark.asyncio
async def test_empty_research_plan_raises_error(researcher):

    with pytest.raises(ValueError):
        await researcher.execute_research_plan([])
