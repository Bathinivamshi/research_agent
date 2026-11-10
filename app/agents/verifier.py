# TODO: Verify that cited evidence supports each claim.

import json
import re

from app.services.llm_service import llm_service


class VerifierAgent:
    def __init__(self, llm=None):
        self.llm = llm or llm_service

    def clean_json_response(self, response: str) -> str:
        response = response.strip()

        return re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            response,
            flags=re.IGNORECASE
        ).strip()

    def validate_quote(self, quote: str, source_content: str) -> bool:
        """Check whether the supporting quote exists in the source."""

        normalize = lambda text: " ".join(text.lower().split())

        if not quote or not source_content:
            return False

        return normalize(quote) in normalize(source_content)

    async def verify_claims(
        self,
        research_question: str,
        evidence: list
    ) -> dict:

        if not research_question or not research_question.strip():
            raise ValueError("Research question cannot be empty")

        if not evidence:
            return {
                "research_question": research_question,
                "total_claims": 0,
                "verified_claims": 0,
                "rejected_claims": 0,
                "verification_passed": False,
                "verified_evidence": [],
                "rejected_evidence": [],
                "message": "No evidence available for verification"
            }

        verified_evidence = []
        rejected_evidence = []

        # First, validate every supporting quote using Python.
        for item in evidence:
            if not isinstance(item, dict):
                continue

            claim = item.get("claim", "").strip()
            quote = item.get("supporting_quote", "").strip()

            if not claim or not quote:
                rejected_evidence.append({
                    **item,
                    "rejection_reason": "Missing claim or supporting quote"
                })
                continue

            source_content = item.get("source_content", "")

            if not source_content:
                rejected_evidence.append({
                    **item,
                    "rejection_reason": "Original source content is unavailable"
                })
                continue

            if not self.validate_quote(quote, source_content):
                rejected_evidence.append({
                    **item,
                    "rejection_reason": "Supporting quote not found in source"
                })
                continue

            verified_evidence.append(item)

        # Do not call Gemini if no quote passed validation.
        if not verified_evidence:
            return {
                "research_question": research_question,
                "total_claims": len(evidence),
                "verified_claims": 0,
                "rejected_claims": len(rejected_evidence),
                "verification_passed": False,
                "verified_evidence": [],
                "rejected_evidence": rejected_evidence,
                "message": "No evidence passed quote validation"
            }

        evidence_for_llm = []

        for index, item in enumerate(verified_evidence):
            evidence_for_llm.append({
                "id": index,
                "claim": item["claim"],
                "supporting_quote": item["supporting_quote"],
                "source_title": item.get("source_title", "")
            })

        prompt = f"""
You are a careful research evidence verifier.

Research question:
{research_question}

Evidence:
{json.dumps(evidence_for_llm, ensure_ascii=False)}

For each evidence item:
1. Decide whether the supporting quote actually supports the claim.
2. Check whether the claim is relevant to the research question.
3. Do not assume a claim is true just because it sounds plausible.
4. Reject claims that exaggerate or misrepresent the quote.
5. Do not use outside knowledge.

Return valid JSON only in this structure:

{{
    "results": [
        {{
            "id": 0,
            "supported": true,
            "reason": "The quote directly supports the claim"
        }}
    ]
}}

Include exactly one result for each evidence ID.
"""

        response = await self.llm.generate_response(prompt)

        try:
            parsed = json.loads(self.clean_json_response(response))
        except (json.JSONDecodeError, TypeError):
            raise ValueError("Verifier returned invalid JSON")

        results = parsed.get("results")

        if not isinstance(results, list):
            raise ValueError("Verifier response must contain a results list")

        results_by_id = {}

        for result in results:
            if not isinstance(result, dict):
                continue

            item_id = result.get("id")

            if (
                isinstance(item_id, int)
                and not isinstance(item_id, bool)
                and 0 <= item_id < len(verified_evidence)
                and item_id not in results_by_id
            ):
                results_by_id[item_id] = result

        final_verified = []

        for index, item in enumerate(verified_evidence):
            result = results_by_id.get(index)

            if result is None:
                rejected_evidence.append({
                    **item,
                    "rejection_reason": "LLM did not return a verification result"
                })
                continue

            if result.get("supported") is True:
                final_verified.append({
                    **item,
                    "verification_reason": str(result.get("reason", "")),
                    "evidence_verified": True
                })
            else:
                rejected_evidence.append({
                    **item,
                    "rejection_reason": str(
                        result.get("reason", "Claim not supported by evidence")
                    )
                })

        verified_count = len(final_verified)

        return {
            "research_question": research_question,
            "total_claims": len(evidence),
            "verified_claims": verified_count,
            "rejected_claims": len(rejected_evidence),
            "verification_passed": verified_count > 0,
            "verified_evidence": final_verified,
            "rejected_evidence": rejected_evidence,
            "message": (
                "Evidence verification completed"
                if verified_count > 0
                else "No claims passed verification"
            )
        }


verifier_agent = VerifierAgent()
