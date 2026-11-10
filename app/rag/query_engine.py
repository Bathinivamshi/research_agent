
from app.rag.ingestion import document_ingestion_service
from app.rag.citations import citation_service
from app.rag.evidence import evidence_verifier
from app.rag.answer_generator import answer_generator


class RAGQueryEngine:

    def __init__(
        self,
        ingestion_service=None,
        citation_service_instance=None,
        evidence_verifier_instance=None,
        answer_generator_instance=None
    ):
        self.ingestion_service = (
            ingestion_service or document_ingestion_service
        )

        self.citation_service = (
            citation_service_instance or citation_service
        )

        self.evidence_verifier = (
            evidence_verifier_instance or evidence_verifier
        )

        self.answer_generator = (
            answer_generator_instance or answer_generator
        )

    # -----------------------------------------
    # Optional RAG evaluation
    # -----------------------------------------

    def _attach_evaluation(
        self,
        result: dict,
        relevant_document_ids: list = None
    ) -> dict:

        # Skip evaluation when reference IDs are not supplied.
        if relevant_document_ids is None:
            return result

        if not isinstance(relevant_document_ids, list):
            raise TypeError(
                "relevant_document_ids must be a list"
            )

        if not relevant_document_ids:
            raise ValueError(
                "Provide at least one relevant document ID for evaluation"
            )

        from app.evaluation.runner import research_evaluation_runner

        result["evaluation"] = (
            research_evaluation_runner.evaluate_rag_result(
                rag_result=result,
                relevant_document_ids=relevant_document_ids
            )
        )

        return result

    # -----------------------------------------
    # Answer user question
    # -----------------------------------------

    async def answer_question(
        self,
        question: str,
        top_k: int = 5,
        relevant_document_ids: list = None
    ) -> dict:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        retrieved_chunks = self.ingestion_service.search_documents(
            query=question,
            top_k=top_k
        )

        # Case 1: No documents retrieved.
        if not retrieved_chunks:

            result = {
                "question": question,
                "answer": "No relevant information found in uploaded documents.",
                "sources": [],
                "evidence": [],
                "retrieved_chunks": [],
                "verified": False
            }

            return self._attach_evaluation(
                result,
                relevant_document_ids
            )

        formatted_results = (
            self.citation_service.format_search_results(
                retrieved_chunks
            )
        )

        verification = self.evidence_verifier.verify_claim(
            claim=question,
            evidence_chunks=retrieved_chunks
        )

        # Case 2: Evidence verification failed.
        if not verification["verified"]:

            result = {
                "question": question,
                "answer": "Insufficient relevant evidence to answer this question.",
                "sources": [],
                "evidence": verification["evidence"],
                "retrieved_chunks": retrieved_chunks,
                "verified": False
            }

            return self._attach_evaluation(
                result,
                relevant_document_ids
            )

        # Case 3: Evidence verified. Generate answer.
        answer = await self.answer_generator.generate_answer(
            question=question,
            evidence_chunks=retrieved_chunks
        )

        result = {
            "question": question,
            "answer": answer,
            "sources": [
                item["citation"]
                for item in formatted_results
            ],
            "evidence": verification["evidence"],
            "retrieved_chunks": retrieved_chunks,
            "verified": True
        }

        return self._attach_evaluation(
            result,
            relevant_document_ids
        )


rag_query_engine = RAGQueryEngine()
