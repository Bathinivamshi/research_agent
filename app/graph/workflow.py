# TODO: Assemble and compile the LangGraph workflow.

from langgraph.graph import StateGraph, START, END

from app.graph.state import ResearchState

from app.graph.nodes import (
    planner_node,
    researcher_node,
    evidence_extractor_node,
    verifier_node,
    report_writer_node
)


def route_after_verification(state: ResearchState) -> str:
    """
    Decide whether the workflow should generate a report
    or stop because the evidence is insufficient.
    """

    if state.get("verification_passed", False):
        return "report_writer"

    return "end"


def build_research_workflow():
    """Build and compile the multi-agent research workflow."""

    workflow = StateGraph(ResearchState)

    # Register agent nodes.
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("evidence_extractor", evidence_extractor_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("report_writer", report_writer_node)

    # Define execution order.
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "evidence_extractor")
    workflow.add_edge("evidence_extractor", "verifier")

    # Route based on verification results.
    workflow.add_conditional_edges(
        "verifier",
        route_after_verification,
        {
            "report_writer": "report_writer",
            "end": END
        }
    )

    workflow.add_edge("report_writer", END)

    return workflow.compile()


class ResearchWorkflow:
    def __init__(self):
        self.graph = build_research_workflow()

    async def run_research(self, research_question: str) -> dict:
        """Execute the complete research workflow."""

        if not research_question or not research_question.strip():
            raise ValueError("Research question cannot be empty")

        initial_state: ResearchState = {
            "research_question": research_question.strip(),
            "documents": [],
            "query_results": [],
            "extracted_evidence": [],
            "verified_evidence": [],
            "rejected_evidence": [],
            "verification_passed": False,
            "verification_summary": {},
            "total_sources_collected": 0,
            "report": {},
            "status": "started",
            "errors": []
        }

        final_state = await self.graph.ainvoke(initial_state)

        return final_state


# Shared workflow instance.
research_workflow = ResearchWorkflow()
