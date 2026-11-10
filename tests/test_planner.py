
import pytest
import json

from app.agents.planner import ResearchPlanner


class MockLLM:

    async def generate_response(self, prompt: str) -> str:

        return json.dumps({
            "topic": "Agentic AI in Healthcare",
            "research_questions": [
                "What are the applications of AI agents in healthcare?",
                "What are the benefits and limitations?",
                "What are the safety and regulatory challenges?"
            ],
            "search_queries": [
                "AI agents applications in healthcare",
                "Benefits and limitations of healthcare AI agents",
                "Healthcare AI agent safety regulations"
            ]
        })


@pytest.fixture
def planner():
    return ResearchPlanner(llm=MockLLM())


@pytest.mark.asyncio
async def test_create_research_plan(planner):

    result = await planner.create_plan(
        "What is the future of agentic AI in healthcare?"
    )

    assert result["topic"] == "Agentic AI in Healthcare"
    assert len(result["research_questions"]) == 3
    assert len(result["search_queries"]) == 3


@pytest.mark.asyncio
async def test_research_plan_contains_valid_queries(planner):

    result = await planner.create_plan(
        "Research AI agents in healthcare"
    )

    for query in result["search_queries"]:
        assert isinstance(query, str)
        assert query.strip()


@pytest.mark.asyncio
async def test_empty_question_raises_error(planner):

    with pytest.raises(ValueError):
        await planner.create_plan("")


class InvalidMockLLM:

    async def generate_response(self, prompt: str) -> str:
        return "This is not valid JSON"


@pytest.mark.asyncio
async def test_invalid_json_raises_error():

    planner = ResearchPlanner(llm=InvalidMockLLM())

    with pytest.raises(ValueError):
        await planner.create_plan("Research AI agents")


class InvalidPlanMockLLM:

    async def generate_response(self, prompt: str) -> str:
        return json.dumps({
            "topic": "AI",
            "research_questions": ["Question 1"],
            "search_queries": ["Query 1"]
        })


@pytest.mark.asyncio
async def test_invalid_plan_structure_raises_error():

    planner = ResearchPlanner(llm=InvalidPlanMockLLM())

    with pytest.raises(ValueError):
        await planner.create_plan("Research AI agents")
