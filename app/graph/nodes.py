# TODO: Implement graph node functions.

from app.graph.state import ResearchState

from app.agents.planner import research_planner
from app.agents.researcher import researcher_agent
from app.agents.evidence_extractor import evidence_extractor_agent
from app.agents.verifier import verifier_agent
from app.agents.report_writer import report_writer_agent


async def planner_node(state: ResearchState) -> dict:
    """Create a research plan from the user's question."""

    question = state["research_question"]

    plan = await research_planner.create_plan(
        research_question=question
    )

    return {
        "plan": plan,
        "status": "planning_completed"
    }


async def researcher_node(state: ResearchState) -> dict:
    """Execute the search queries and collect source documents."""

    plan = state["plan"]
    search_queries = plan["search_queries"]

    research_results = await researcher_agent.execute_research_plan(
        search_queries=search_queries
    )

    return {
        "documents": research_results["documents"],
        "query_results": research_results["query_results"],
        "total_sources_collected": research_results[
            "total_sources_collected"
        ],
        "status": "research_completed"
    }


async def evidence_extractor_node(state: ResearchState) -> dict:
    """Extract claims and supporting quotes from collected documents."""

    documents = state.get("documents", [])

    extraction_result = await evidence_extractor_agent.extract_evidence(
        research_question=state["research_question"],
        documents=documents
    )

    return {
        "extracted_evidence": extraction_result["evidence"],
        "status": "evidence_extraction_completed"
    }


async def verifier_node(state: ResearchState) -> dict:
    """Verify extracted claims against their original source content."""

    evidence = state.get("extracted_evidence", [])

    verification_result = await verifier_agent.verify_claims(
        research_question=state["research_question"],
        evidence=evidence
    )

    return {
        "verified_evidence": verification_result["verified_evidence"],
        "rejected_evidence": verification_result["rejected_evidence"],
        "verification_passed": verification_result["verification_passed"],
        "verification_summary": verification_result,
        "status": "verification_completed"
    }


async def report_writer_node(state: ResearchState) -> dict:
    """Generate the final report using verified evidence."""

    report = await report_writer_agent.generate_report(
        research_question=state["research_question"],
        verified_evidence=state.get("verified_evidence", []),
        rejected_evidence=state.get("rejected_evidence", [])
    )

    return {
        "report": report,
        "status": "report_completed"
    }
