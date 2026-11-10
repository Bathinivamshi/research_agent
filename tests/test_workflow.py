
import pytest

from app.graph.state import ResearchState
from app.graph.workflow import (
    build_research_workflow,
    route_after_verification
)


def test_workflow_compiles():
    graph = build_research_workflow()

    assert graph is not None


def test_successful_verification_routes_to_report_writer():
    state: ResearchState = {
        "verification_passed": True
    }

    result = route_after_verification(state)

    assert result == "report_writer"


def test_failed_verification_routes_to_end():
    state: ResearchState = {
        "verification_passed": False
    }

    result = route_after_verification(state)

    assert result == "end"


def test_missing_verification_routes_to_end():
    state: ResearchState = {}

    result = route_after_verification(state)

    assert result == "end"


@pytest.mark.asyncio
async def test_empty_question_is_rejected():
    from app.graph.workflow import research_workflow

    with pytest.raises(ValueError):
        await research_workflow.run_research("")
