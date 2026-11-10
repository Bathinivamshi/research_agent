
class ResearchMetrics:
    """Deterministic metrics for evaluating research and RAG systems."""

    @staticmethod
    def _validate_count(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} cannot be negative")

    @staticmethod
    def _validate_ids(values: list, name: str) -> None:
        if not isinstance(values, list):
            raise TypeError(f"{name} must be a list")

        if any(not isinstance(value, str) or not value.strip()
               for value in values):
            raise ValueError(f"{name} must contain non-empty strings")

    @staticmethod
    def calculate_precision(
        relevant_retrieved: int,
        total_retrieved: int
    ) -> float:
        ResearchMetrics._validate_count(relevant_retrieved, "relevant_retrieved")
        ResearchMetrics._validate_count(total_retrieved, "total_retrieved")

        if relevant_retrieved > total_retrieved:
            raise ValueError("Relevant retrieved cannot exceed total retrieved")

        if total_retrieved == 0:
            return 0.0

        return round(relevant_retrieved / total_retrieved, 4)

    @staticmethod
    def calculate_recall(
        relevant_retrieved: int,
        total_relevant: int
    ) -> float:
        ResearchMetrics._validate_count(relevant_retrieved, "relevant_retrieved")
        ResearchMetrics._validate_count(total_relevant, "total_relevant")

        if relevant_retrieved > total_relevant:
            raise ValueError("Relevant retrieved cannot exceed total relevant")

        if total_relevant == 0:
            return 0.0

        return round(relevant_retrieved / total_relevant, 4)

    @staticmethod
    def calculate_f1(precision: float, recall: float) -> float:
        if not 0 <= precision <= 1 or not 0 <= recall <= 1:
            raise ValueError("Precision and recall must be between 0 and 1")

        if precision + recall == 0:
            return 0.0

        return round(2 * precision * recall / (precision + recall), 4)

    @staticmethod
    def calculate_citation_accuracy(
        valid_citations: int,
        total_citations: int
    ) -> float:
        ResearchMetrics._validate_count(valid_citations, "valid_citations")
        ResearchMetrics._validate_count(total_citations, "total_citations")

        if valid_citations > total_citations:
            raise ValueError("Valid citations cannot exceed total citations")

        if total_citations == 0:
            return 0.0

        return round(valid_citations / total_citations, 4)

    @staticmethod
    def calculate_evidence_coverage(
        supported_claims: int,
        total_claims: int
    ) -> float:
        ResearchMetrics._validate_count(supported_claims, "supported_claims")
        ResearchMetrics._validate_count(total_claims, "total_claims")

        if supported_claims > total_claims:
            raise ValueError("Supported claims cannot exceed total claims")

        if total_claims == 0:
            return 0.0

        return round(supported_claims / total_claims, 4)

    @staticmethod
    def calculate_quote_accuracy(
        valid_quotes: int,
        total_quotes: int
    ) -> float:
        ResearchMetrics._validate_count(valid_quotes, "valid_quotes")
        ResearchMetrics._validate_count(total_quotes, "total_quotes")

        if valid_quotes > total_quotes:
            raise ValueError("Valid quotes cannot exceed total quotes")

        if total_quotes == 0:
            return 0.0

        return round(valid_quotes / total_quotes, 4)

    @staticmethod
    def evaluate_retrieval(
        retrieved_document_ids: list,
        relevant_document_ids: list
    ) -> dict:
        ResearchMetrics._validate_ids(
            retrieved_document_ids, "retrieved_document_ids"
        )
        ResearchMetrics._validate_ids(
            relevant_document_ids, "relevant_document_ids"
        )

        retrieved = set(retrieved_document_ids)
        relevant = set(relevant_document_ids)

        relevant_retrieved = len(retrieved.intersection(relevant))

        precision = ResearchMetrics.calculate_precision(
            relevant_retrieved, len(retrieved)
        )
        recall = ResearchMetrics.calculate_recall(
            relevant_retrieved, len(relevant)
        )
        f1 = ResearchMetrics.calculate_f1(precision, recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "relevant_retrieved": relevant_retrieved,
            "total_retrieved": len(retrieved),
            "total_relevant": len(relevant)
        }

    @staticmethod
    def evaluate_citations(
        citations: list,
        available_source_ids: list
    ) -> dict:
        ResearchMetrics._validate_ids(citations, "citations")
        ResearchMetrics._validate_ids(
            available_source_ids, "available_source_ids"
        )

        available = set(available_source_ids)
        valid = [citation for citation in citations if citation in available]

        total = len(citations)
        accuracy = ResearchMetrics.calculate_citation_accuracy(
            len(valid), total
        )

        return {
            "citation_accuracy": accuracy,
            "valid_citations": len(valid),
            "invalid_citations": total - len(valid),
            "total_citations": total
        }

    @staticmethod
    def generate_evaluation_report(
        retrieval_metrics: dict,
        citation_metrics: dict,
        evidence_coverage: float,
        quote_accuracy: float
    ) -> dict:
        return {
            "retrieval": retrieval_metrics,
            "citations": citation_metrics,
            "evidence_coverage": evidence_coverage,
            "quote_accuracy": quote_accuracy
        }


research_metrics = ResearchMetrics()
