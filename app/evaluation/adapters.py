
class EvaluationAdapter:

    @staticmethod
    def adapt_rag_result(result: dict) -> dict:
        """Normalize actual RAG output for evaluation."""

        if not isinstance(result, dict):
            raise TypeError("RAG result must be a dictionary")

        retrieved_chunks = result.get("retrieved_chunks", [])
        evidence = result.get("evidence", [])
        sources = result.get("sources", [])

        retrieved_ids = list(dict.fromkeys(
            str(chunk["document_id"])
            for chunk in retrieved_chunks
            if chunk.get("document_id") is not None
        ))

        citation_ids = list(dict.fromkeys(
            str(source["document_id"])
            for source in sources
            if source.get("document_id") is not None
        ))

        supported_claims = sum(
            1 for item in evidence
            if item.get("supports_claim") is True
        )

        return {
            "question": result.get("question", ""),
            "answer": result.get("answer", ""),
            "retrieved_document_ids": retrieved_ids,
            "citations": citation_ids,
            "supported_claims": supported_claims,
            "total_claims": len(evidence),
            "valid_quotes": 0,
            "total_quotes": 0,
            "verified": result.get("verified", False),
            "evaluation_type": "rag"
        }

    @staticmethod
    def adapt_agentic_result(result: dict) -> dict:
        """Normalize actual Agentic Research output for evaluation."""

        if not isinstance(result, dict):
            raise TypeError("Agentic result must be a dictionary")

        verified = result.get("verified_evidence", [])
        rejected = result.get("rejected_evidence", [])
        report = result.get("report", {})

        all_evidence = verified + rejected

        retrieved_ids = list(dict.fromkeys(
            str(item["source_id"])
            for item in all_evidence
            if item.get("source_id")
        ))

        report_sources = (
            report.get("sources", [])
            if isinstance(report, dict) else []
        )

        citation_ids = list(dict.fromkeys(
            str(item["source_id"])
            for item in report_sources
            if item.get("source_id")
        ))

        valid_quotes = sum(
            1 for item in verified
            if item.get("evidence_verified") is True
        )

        return {
            "question": result.get("research_question", ""),
            "answer": report,
            "retrieved_document_ids": retrieved_ids,
            "citations": citation_ids,
            "supported_claims": len(verified),
            "total_claims": len(all_evidence),
            "valid_quotes": valid_quotes,
            "total_quotes": len(all_evidence),
            "verified": result.get("verification_passed", False),
            "evaluation_type": "agentic"
        }


evaluation_adapter = EvaluationAdapter()
