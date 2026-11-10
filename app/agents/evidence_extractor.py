
# TODO: Extract atomic claims and supporting passages.

import json
import re

from app.services.llm_service import llm_service


class EvidenceExtractorAgent:

    def __init__(self, llm=None):
        self.llm = llm or llm_service
        self.max_content_length = 12000

    def clean_json_response(self, response: str) -> str:
        """Remove Markdown code fences from an LLM response."""

        response = response.strip()

        response = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            response,
            flags=re.IGNORECASE
        )

        return response.strip()

    def validate_quote(
        self,
        quote: str,
        original_content: str
    ) -> bool:
        """Check whether the extracted quote exists in the original source."""

        def normalize(text):
            return " ".join(text.lower().split())

        if not quote or not original_content:
            return False

        return normalize(quote) in normalize(original_content)

    async def extract_from_document(
        self,
        research_question: str,
        document: dict
    ) -> list:

        title = document.get("title", "Unknown source")
        url = document.get("url", "")
        source_id = document.get("source_id")
        content = document.get("content", "")

        if not content or not content.strip():
            return []

        original_content = content.strip()
        limited_content = original_content[:self.max_content_length]

        prompt = f"""
You are an evidence extraction assistant.

Research question:
{research_question}

Source title:
{title}

Source URL:
{url}

Source content:
{limited_content}

Your task:
1. Extract factual claims relevant to the research question.
2. For every claim, provide an exact supporting quote from the source.
3. Do not invent facts or quotes.
4. Ignore irrelevant information.
5. Return at most 8 evidence items.
6. Return valid JSON only.

Required JSON structure:
{{
    "evidence": [
        {{
            "claim": "A concise factual claim",
            "supporting_quote": "An exact sentence or passage from the source"
        }}
    ]
}}
"""

        response = await self.llm.generate_response(prompt)

        try:
            cleaned_response = self.clean_json_response(response)
            parsed_response = json.loads(cleaned_response)

        except (json.JSONDecodeError, TypeError):
            raise ValueError(
                f"Evidence extraction returned invalid JSON for: {title}"
            )

        extracted_items = parsed_response.get("evidence")

        if not isinstance(extracted_items, list):
            raise ValueError(
                "Evidence response must contain an evidence list"
            )

        verified_evidence = []

        for item in extracted_items:

            if not isinstance(item, dict):
                continue

            claim = item.get("claim", "")
            quote = item.get("supporting_quote", "")

            if not isinstance(claim, str) or not claim.strip():
                continue

            if not isinstance(quote, str) or not quote.strip():
                continue

            if not self.validate_quote(quote, original_content):
                continue

            verified_evidence.append({
                "claim": claim.strip(),
                "supporting_quote": quote.strip(),

                # Preserve source identity.
                "source_id": source_id,

                "source_title": title,
                "source_url": url,
                "source_content": original_content,
                "page_number": document.get("page_number"),
                "evidence_verified": True
            })

        return verified_evidence

    async def extract_evidence(
        self,
        research_question: str,
        documents: list
    ) -> dict:

        if not research_question or not research_question.strip():
            raise ValueError("Research question cannot be empty")

        if not documents:
            raise ValueError(
                "No documents provided for evidence extraction"
            )

        all_evidence = []
        failed_documents = []

        for document in documents:

            try:
                evidence = await self.extract_from_document(
                    research_question=research_question,
                    document=document
                )

                all_evidence.extend(evidence)

            except Exception as error:
                failed_documents.append({
                    "source_id": document.get("source_id"),
                    "title": document.get("title", "Unknown source"),
                    "error": str(error)
                })

        return {
            "research_question": research_question,
            "total_documents": len(documents),
            "total_evidence": len(all_evidence),
            "failed_documents": failed_documents,
            "evidence": all_evidence
        }


evidence_extractor_agent = EvidenceExtractorAgent()
