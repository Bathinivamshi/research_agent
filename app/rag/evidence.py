
from app.rag.embeddings import embedding_service


class EvidenceVerifier:

    def __init__(self, minimum_similarity: float = 0.40):

        if not 0 <= minimum_similarity <= 1:
            raise ValueError(
                "Minimum similarity must be between 0 and 1"
            )

        self.minimum_similarity = minimum_similarity

    def verify_claim(
        self,
        claim: str,
        evidence_chunks: list[dict]
    ) -> dict:

        if not claim or not claim.strip():
            raise ValueError("Claim cannot be empty")

        if not evidence_chunks:
            return {
                "claim": claim,
                "verified": False,
                "reason": "No evidence provided",
                "evidence": []
            }

        claim_embedding = (
            embedding_service.generate_embedding(claim)
        )

        verified_evidence = []

        for chunk in evidence_chunks:

            evidence_text = chunk.get("text", "")

            if not evidence_text.strip():
                continue

            evidence_embedding = (
                embedding_service.generate_embedding(evidence_text)
            )

            similarity = (
                embedding_service.cosine_similarity(
                    claim_embedding,
                    evidence_embedding
                )
            )

            supports_claim = (
                similarity >= self.minimum_similarity
            )

            verified_evidence.append({
                **chunk,
                "claim_similarity": similarity,
                "supports_claim": supports_claim
            })

        supporting_evidence = [
            item for item in verified_evidence
            if item["supports_claim"]
        ]

        return {
            "claim": claim,
            "verified": len(supporting_evidence) > 0,
            "reason": (
                "Supporting evidence found"
                if supporting_evidence
                else "No sufficiently similar evidence found"
            ),
            "evidence": verified_evidence
        }


evidence_verifier = EvidenceVerifier()
