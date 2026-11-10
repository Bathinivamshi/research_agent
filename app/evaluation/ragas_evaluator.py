
import asyncio
from typing import Any

from ragas import SingleTurnSample
from ragas.metrics import (
    IDBasedContextPrecision,
    IDBasedContextRecall,
)


class RagasEvaluator:

    def __init__(self):
        self.precision_metric = IDBasedContextPrecision()
        self.recall_metric = IDBasedContextRecall()

    @staticmethod
    def validate_ids(ids: list, name: str) -> None:
        if not isinstance(ids, list):
            raise TypeError(f"{name} must be a list")

        if not ids:
            raise ValueError(f"{name} cannot be empty")

        for item in ids:
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                raise ValueError(
                    f"{name} must contain only strings or integers"
                )

    async def evaluate_retrieval(
        self,
        retrieved_context_ids: list,
        reference_context_ids: list,
    ) -> dict[str, Any]:

        self.validate_ids(
            retrieved_context_ids,
            "retrieved_context_ids"
        )
        self.validate_ids(
            reference_context_ids,
            "reference_context_ids"
        )

        # Normalize IDs and remove duplicates
        retrieved_ids = list(
            dict.fromkeys(str(item) for item in retrieved_context_ids)
        )
        reference_ids = list(
            dict.fromkeys(str(item) for item in reference_context_ids)
        )

        # Create Ragas evaluation sample
        sample = SingleTurnSample(
            retrieved_context_ids=retrieved_ids,
            reference_context_ids=reference_ids,
        )

        # Calculate precision and recall concurrently
        precision, recall = await asyncio.gather(
            self.precision_metric.single_turn_ascore(sample),
            self.recall_metric.single_turn_ascore(sample),
        )

        # Calculate F1 score
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = (
                2 * precision * recall
            ) / (precision + recall)

        # Identify correctly retrieved documents
        reference_set = set(reference_ids)

        matched_ids = [
            doc_id
            for doc_id in retrieved_ids
            if doc_id in reference_set
        ]

        return {
            "context_precision": round(float(precision), 4),
            "context_recall": round(float(recall), 4),
            "f1_score": round(float(f1_score), 4),
            "total_retrieved": len(retrieved_ids),
            "total_relevant": len(reference_ids),
            "total_matched": len(matched_ids),
            "matched_ids": matched_ids,
        }
