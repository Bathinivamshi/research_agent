
import pytest

from app.evaluation.ragas_evaluator import RagasEvaluator


@pytest.mark.asyncio
async def test_retrieval_evaluation():

    evaluator = RagasEvaluator()

    retrieved_ids = [
        "doc_1",
        "doc_2",
        "doc_3",
        "doc_4",
    ]

    reference_ids = [
        "doc_1",
        "doc_3",
        "doc_5",
        "doc_6",
    ]

    result = await evaluator.evaluate_retrieval(
        retrieved_context_ids=retrieved_ids,
        reference_context_ids=reference_ids,
    )

    assert result["context_precision"] == pytest.approx(0.5)
    assert result["context_recall"] == pytest.approx(0.5)
    assert result["f1_score"] == pytest.approx(0.5)

    assert result["total_retrieved"] == 4
    assert result["total_relevant"] == 4
    assert result["total_matched"] == 2


@pytest.mark.asyncio
async def test_empty_retrieved_ids():

    evaluator = RagasEvaluator()

    with pytest.raises(ValueError):
        await evaluator.evaluate_retrieval(
            retrieved_context_ids=[],
            reference_context_ids=["doc_1"],
        )


@pytest.mark.asyncio
async def test_invalid_ids():

    evaluator = RagasEvaluator()

    with pytest.raises(TypeError):
        await evaluator.evaluate_retrieval(
            retrieved_context_ids="doc_1",
            reference_context_ids=["doc_1"],
        )
