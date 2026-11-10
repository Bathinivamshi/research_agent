
# Define the shared LangGraph state schema.

from typing import TypedDict, Any


class ResearchState(TypedDict, total=False):

    # Original research request
    research_question: str

    # Planner output
    plan: dict

    # Researcher output
    documents: list
    query_results: list
    total_sources_collected: int

    # Evidence extractor output
    extracted_evidence: list

    # Verifier output
    verified_evidence: list
    rejected_evidence: list
    verification_passed: bool
    verification_summary: dict

    # Report writer output
    report: dict

    # Evaluation input data
    retrieved_document_ids: list
    citations: list
    supported_claims: int
    total_claims: int
    valid_quotes: int
    total_quotes: int

    # Evaluation results
    evaluation_results: dict
    evaluation_report_path: str

    # Workflow control
    retry_count: int
    max_retries: int
    next_action: str
    status: str
    errors: list
