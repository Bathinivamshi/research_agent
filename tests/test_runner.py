
import pytest

from app.evaluation.runner import research_evaluation_runner


def test_evaluate_single_case():
    result = research_evaluation_runner.evaluate_case(
        case_id="test_001",
        retrieved_document_ids=["doc_001", "doc_002"],
        citations=["source_001"],
        supported_claims=8,
        total_claims=10,
        valid_quotes=9,
        total_quotes=10
    )

    assert result["retrieval"]["precision"] == 1.0
    assert result["retrieval"]["recall"] == 1.0
    assert result["citations"]["citation_accuracy"] == 1.0
    assert result["evidence_coverage"] == 0.8
    assert result["quote_accuracy"] == 0.9


def test_evaluate_complete_dataset():
    results = research_evaluation_runner.evaluate_dataset([
        {
            "case_id": "test_001",
            "retrieved_document_ids": ["doc_001", "doc_002"],
            "citations": ["source_001"],
            "supported_claims": 8,
            "total_claims": 10,
            "valid_quotes": 9,
            "total_quotes": 10
        },
        {
            "case_id": "test_002",
            "retrieved_document_ids": ["doc_003"],
            "citations": ["source_003"],
            "supported_claims": 5,
            "total_claims": 5,
            "valid_quotes": 5,
            "total_quotes": 5
        }
    ])

    assert results["total_evaluated_cases"] == 2
    assert results["average_precision"] == 1.0
    assert results["average_recall"] == 0.75


def test_unknown_case_raises_error():
    with pytest.raises(ValueError):
        research_evaluation_runner.evaluate_case(
            case_id="unknown",
            retrieved_document_ids=[],
            citations=[],
            supported_claims=0,
            total_claims=0,
            valid_quotes=0,
            total_quotes=0
        )


def test_empty_dataset_results_raise_error():
    with pytest.raises(ValueError):
        research_evaluation_runner.evaluate_dataset([])


def test_duplicate_cases_are_rejected():
    case = {
        "case_id": "test_001",
        "retrieved_document_ids": [],
        "citations": [],
        "supported_claims": 0,
        "total_claims": 0,
        "valid_quotes": 0,
        "total_quotes": 0
    }

    with pytest.raises(ValueError):
        research_evaluation_runner.evaluate_dataset([case, case])
