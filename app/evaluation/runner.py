
import asyncio
import json
import uuid

from pathlib import Path
from datetime import datetime

from app.evaluation.metrics import research_metrics
from app.evaluation.test_dataset import research_test_dataset
from app.evaluation.adapters import evaluation_adapter


class ResearchEvaluationRunner:

    def __init__(self):
        self.dataset = research_test_dataset
        self.metrics = research_metrics
        self.ragas_evaluator = None

    # -----------------------------------------
    # Lazy load Ragas only when required
    # -----------------------------------------

    def _get_ragas_evaluator(self):
        if self.ragas_evaluator is None:
            from app.evaluation.ragas_evaluator import RagasEvaluator
            self.ragas_evaluator = RagasEvaluator()

        return self.ragas_evaluator

    # -----------------------------------------
    # Evaluate one benchmark case
    # -----------------------------------------

    def evaluate_case(
        self,
        case_id: str,
        retrieved_document_ids: list,
        citations: list,
        supported_claims: int,
        total_claims: int,
        valid_quotes: int,
        total_quotes: int
    ) -> dict:

        case = self.dataset.get_case_by_id(case_id)

        if case is None:
            raise ValueError(f"Unknown evaluation case: {case_id}")

        retrieval = self.metrics.evaluate_retrieval(
            retrieved_document_ids=retrieved_document_ids,
            relevant_document_ids=case["relevant_document_ids"]
        )

        citation_results = self.metrics.evaluate_citations(
            citations=citations,
            available_source_ids=case["expected_sources"]
        )

        evidence_coverage = self.metrics.calculate_evidence_coverage(
            supported_claims,
            total_claims
        )

        quote_accuracy = self.metrics.calculate_quote_accuracy(
            valid_quotes,
            total_quotes
        )

        return {
            "case_id": case_id,
            "category": case["category"],
            "question": case["question"],
            "retrieval": retrieval,
            "citations": citation_results,
            "evidence_coverage": evidence_coverage,
            "quote_accuracy": quote_accuracy
        }

    # -----------------------------------------
    # Evaluate one case with Ragas
    # -----------------------------------------

    async def evaluate_case_with_ragas(
        self,
        case_id: str,
        retrieved_document_ids: list,
        citations: list,
        supported_claims: int,
        total_claims: int,
        valid_quotes: int,
        total_quotes: int
    ) -> dict:

        result = self.evaluate_case(
            case_id=case_id,
            retrieved_document_ids=retrieved_document_ids,
            citations=citations,
            supported_claims=supported_claims,
            total_claims=total_claims,
            valid_quotes=valid_quotes,
            total_quotes=total_quotes
        )

        case = self.dataset.get_case_by_id(case_id)

        ragas_results = await self._get_ragas_evaluator().evaluate_retrieval(
            retrieved_context_ids=retrieved_document_ids,
            reference_context_ids=case["relevant_document_ids"]
        )

        result["ragas_retrieval"] = ragas_results

        return result

    # -----------------------------------------
    # Evaluate complete benchmark dataset
    # -----------------------------------------

    def evaluate_dataset(
        self,
        evaluation_results: list
    ) -> dict:

        if not isinstance(evaluation_results, list) or not evaluation_results:
            raise ValueError("Evaluation results must be a non-empty list")

        case_results = []
        seen_case_ids = set()

        for result in evaluation_results:

            if not isinstance(result, dict):
                raise ValueError(
                    "Each evaluation result must be a dictionary"
                )

            case_id = result.get("case_id")

            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(
                    "Every evaluation result needs a valid case_id"
                )

            if case_id in seen_case_ids:
                raise ValueError(
                    f"Duplicate evaluation case: {case_id}"
                )

            seen_case_ids.add(case_id)

            case_results.append(
                self.evaluate_case(
                    case_id=case_id,
                    retrieved_document_ids=result.get(
                        "retrieved_document_ids", []
                    ),
                    citations=result.get("citations", []),
                    supported_claims=result.get("supported_claims", 0),
                    total_claims=result.get("total_claims", 0),
                    valid_quotes=result.get("valid_quotes", 0),
                    total_quotes=result.get("total_quotes", 0)
                )
            )

        count = len(case_results)

        def average(values):
            return round(sum(values) / len(values), 4) if values else 0.0

        return {
            "total_evaluated_cases": count,

            "average_precision": average([
                item["retrieval"]["precision"]
                for item in case_results
            ]),

            "average_recall": average([
                item["retrieval"]["recall"]
                for item in case_results
            ]),

            "average_f1_score": average([
                item["retrieval"]["f1_score"]
                for item in case_results
            ]),

            "average_citation_accuracy": average([
                item["citations"]["citation_accuracy"]
                for item in case_results
            ]),

            "average_evidence_coverage": average([
                item["evidence_coverage"]
                for item in case_results
            ]),

            "average_quote_accuracy": average([
                item["quote_accuracy"]
                for item in case_results
            ]),

            "case_results": case_results
        }

    # -----------------------------------------
    # Evaluate complete dataset with Ragas
    # -----------------------------------------

    async def evaluate_dataset_with_ragas(
        self,
        evaluation_results: list
    ) -> dict:

        if not isinstance(evaluation_results, list) or not evaluation_results:
            raise ValueError("Evaluation results must be a non-empty list")

        report = self.evaluate_dataset(evaluation_results)

        semaphore = asyncio.Semaphore(3)

        async def evaluate_one(result):
            async with semaphore:
                return await self.evaluate_case_with_ragas(
                    case_id=result["case_id"],
                    retrieved_document_ids=result.get(
                        "retrieved_document_ids", []
                    ),
                    citations=result.get("citations", []),
                    supported_claims=result.get("supported_claims", 0),
                    total_claims=result.get("total_claims", 0),
                    valid_quotes=result.get("valid_quotes", 0),
                    total_quotes=result.get("total_quotes", 0)
                )

        ragas_case_results = await asyncio.gather(
            *(evaluate_one(result) for result in evaluation_results)
        )

        def average(values):
            return round(sum(values) / len(values), 4) if values else 0.0

        report["average_ragas_context_precision"] = average([
            item["ragas_retrieval"]["context_precision"]
            for item in ragas_case_results
        ])

        report["average_ragas_context_recall"] = average([
            item["ragas_retrieval"]["context_recall"]
            for item in ragas_case_results
        ])

        report["average_ragas_f1_score"] = average([
            item["ragas_retrieval"]["f1_score"]
            for item in ragas_case_results
        ])

        report["ragas_case_results"] = ragas_case_results

        return report

    # -----------------------------------------
    # Evaluate actual live application output
    # -----------------------------------------

    def evaluate_live_result(
        self,
        retrieved_document_ids: list,
        relevant_document_ids: list,
        citations: list,
        expected_source_ids: list,
        supported_claims: int,
        total_claims: int,
        valid_quotes: int,
        total_quotes: int,
        question: str = "",
        evaluation_type: str = "rag"
    ) -> dict:

        if not isinstance(evaluation_type, str) or evaluation_type not in {
            "rag", "agentic"
        }:
            raise ValueError("evaluation_type must be 'rag' or 'agentic'")

        retrieval = self.metrics.evaluate_retrieval(
            retrieved_document_ids=retrieved_document_ids,
            relevant_document_ids=relevant_document_ids
        )

        citation_results = self.metrics.evaluate_citations(
            citations=citations,
            available_source_ids=expected_source_ids
        )

        evidence_coverage = self.metrics.calculate_evidence_coverage(
            supported_claims,
            total_claims
        )

        quote_accuracy = self.metrics.calculate_quote_accuracy(
            valid_quotes,
            total_quotes
        )

        return {
            "evaluation_type": evaluation_type,
            "question": question,
            "retrieval": retrieval,
            "citations": citation_results,
            "evidence_coverage": evidence_coverage,
            "quote_accuracy": quote_accuracy,
            "evaluation_source": "live_application_output"
        }

    # -----------------------------------------
    # Evaluate adapted application output
    # -----------------------------------------

    def evaluate_adapted_result(
        self,
        adapted_result: dict,
        relevant_document_ids: list,
        expected_source_ids: list
    ) -> dict:

        if not isinstance(adapted_result, dict):
            raise TypeError("Adapted result must be a dictionary")

        required_fields = [
            "retrieved_document_ids",
            "citations",
            "supported_claims",
            "total_claims",
            "valid_quotes",
            "total_quotes"
        ]

        missing_fields = [
            field for field in required_fields
            if field not in adapted_result
        ]

        if missing_fields:
            raise ValueError(
                f"Missing adapter fields: {missing_fields}"
            )

        return self.evaluate_live_result(
            retrieved_document_ids=adapted_result["retrieved_document_ids"],
            relevant_document_ids=relevant_document_ids,
            citations=adapted_result["citations"],
            expected_source_ids=expected_source_ids,
            supported_claims=adapted_result["supported_claims"],
            total_claims=adapted_result["total_claims"],
            valid_quotes=adapted_result["valid_quotes"],
            total_quotes=adapted_result["total_quotes"],
            question=adapted_result.get("question", ""),
            evaluation_type=adapted_result.get("evaluation_type", "rag")
        )

    # -----------------------------------------
    # Evaluate actual RAG query result
    # -----------------------------------------

    def evaluate_rag_result(
        self,
        rag_result: dict,
        relevant_document_ids: list
    ) -> dict:

        adapted_result = evaluation_adapter.adapt_rag_result(
            rag_result
        )

        return self.evaluate_adapted_result(
            adapted_result=adapted_result,
            relevant_document_ids=relevant_document_ids,
            expected_source_ids=relevant_document_ids
        )

    # -----------------------------------------
    # Save evaluation report
    # -----------------------------------------

    def save_report(self, report: dict) -> str:

        if not isinstance(report, dict):
            raise TypeError("Report must be a dictionary")

        reports_dir = Path(__file__).parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_id = uuid.uuid4().hex[:8]

        filename = (
            f"evaluation_report_{timestamp}_{unique_id}.json"
        )

        file_path = reports_dir / filename

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False
            )

        return str(file_path)


research_evaluation_runner = ResearchEvaluationRunner()
