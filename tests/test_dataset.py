
from app.evaluation.test_dataset import research_test_dataset


def test_dataset_contains_five_cases():
    cases = research_test_dataset.get_all_cases()

    assert len(cases) == 5


def test_dataset_validation():
    assert research_test_dataset.validate_dataset() is True


def test_get_case_by_id():
    case = research_test_dataset.get_case_by_id("test_001")

    assert case is not None
    assert case["category"] == "RAG"


def test_get_nonexistent_case():
    case = research_test_dataset.get_case_by_id("test_999")

    assert case is None


def test_filter_by_category():
    cases = research_test_dataset.get_cases_by_category("Agentic AI")

    assert len(cases) == 1
    assert cases[0]["id"] == "test_002"


def test_dataset_summary():
    summary = research_test_dataset.get_dataset_summary()

    assert summary["total_cases"] == 5
    assert "RAG" in summary["categories"]
