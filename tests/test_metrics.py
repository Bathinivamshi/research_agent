
from app.evaluation.metrics import research_metrics


def test_precision():
    result = research_metrics.calculate_precision(8, 10)
    assert result == 0.8


def test_recall():
    result = research_metrics.calculate_recall(8, 10)
    assert result == 0.8


def test_f1_score():
    result = research_metrics.calculate_f1(0.8, 0.8)
    assert result == 0.8


def test_retrieval_evaluation():
    result = research_metrics.evaluate_retrieval(
        retrieved_document_ids=["A", "B", "C"],
        relevant_document_ids=["A", "B", "D"]
    )

    assert result["precision"] == 0.6667
    assert result["recall"] == 0.6667


def test_citation_accuracy():
    result = research_metrics.evaluate_citations(
        citations=["S1", "S2", "S99"],
        available_source_ids=["S1", "S2", "S3"]
    )

    assert result["citation_accuracy"] == 0.6667
    assert result["invalid_citations"] == 1


def test_zero_division():
    assert research_metrics.calculate_precision(0, 0) == 0.0
    assert research_metrics.calculate_recall(0, 0) == 0.0


def test_evidence_coverage():
    result = research_metrics.calculate_evidence_coverage(8, 10)
    assert result == 0.8


def test_quote_accuracy():
    result = research_metrics.calculate_quote_accuracy(9, 10)
    assert result == 0.9
