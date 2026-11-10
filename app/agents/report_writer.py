
# TODO: Generate the structured final report with citations.

import json
import re

from app.services.llm_service import llm_service


class ReportWriterAgent:

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

    async def generate_report(
        self,
        research_question: str,
        verified_evidence: list,
        rejected_evidence: list | None = None
    ) -> dict:

        if not research_question or not research_question.strip():
            raise ValueError("Research question cannot be empty")

        if not verified_evidence:
            raise ValueError(
                "Cannot generate a research report without verified evidence"
            )

        evidence_for_report = []

        for index, item in enumerate(verified_evidence, start=1):

            evidence_for_report.append({
                "citation_id": f"S{index}",

                # Preserve original source identity.
                "source_id": item.get("source_id"),

                "claim": item.get("claim", ""),
                "supporting_quote": item.get("supporting_quote", ""),
                "source_title": item.get("source_title", "Unknown source"),
                "source_url": item.get("source_url", ""),
                "page_number": item.get("page_number"),
                "verification_reason": item.get(
                    "verification_reason", ""
                )
            })

        prompt = f"""
You are a professional research report writer.

Research question:
{research_question}

Verified evidence:
{json.dumps(evidence_for_report, ensure_ascii=False)}

Instructions:
1. Use only the supplied verified evidence.
2. Do not introduce outside facts or unsupported statistics.
3. Organize the report into a clear introduction, key findings,
   analysis, limitations, and conclusion.
4. Cite supporting findings using the supplied citation IDs,
   such as [S1] or [S2].
5. Do not invent sources, URLs, page numbers, or quotations.
6. If evidence is insufficient for a conclusion, say so clearly.
7. Write in professional, readable language.

Return valid JSON only with this structure:

{{
    "title": "Research report title",
    "executive_summary": "A concise summary",
    "introduction": "Introduction",
    "key_findings": [
        {{
            "finding": "A supported finding",
            "citations": ["S1"]
        }}
    ],
    "analysis": "Detailed analysis with inline citation IDs",
    "limitations": ["Research limitation"],
    "conclusion": "Evidence-based conclusion"
}}
"""

        response = await self.llm.generate_response(prompt)

        try:
            report = json.loads(
                self.clean_json_response(response)
            )
        except (json.JSONDecodeError, TypeError):
            raise ValueError("Report writer returned invalid JSON")

        if not isinstance(report, dict):
            raise ValueError("Report response must be a JSON object")

        required_fields = [
            "title",
            "executive_summary",
            "introduction",
            "key_findings",
            "analysis",
            "limitations",
            "conclusion"
        ]

        for field in required_fields:
            if field not in report:
                raise ValueError(
                    f"Report is missing required field: {field}"
                )

        text_fields = [
            "title",
            "executive_summary",
            "introduction",
            "analysis",
            "conclusion"
        ]

        for field in text_fields:
            if (
                not isinstance(report[field], str)
                or not report[field].strip()
            ):
                raise ValueError(
                    f"Report field '{field}' must be non-empty text"
                )

        if not isinstance(report["key_findings"], list):
            raise ValueError("key_findings must be a list")

        if not isinstance(report["limitations"], list):
            raise ValueError("limitations must be a list")

        allowed_citations = {
            item["citation_id"]
            for item in evidence_for_report
        }

        for finding in report["key_findings"]:

            if not isinstance(finding, dict):
                raise ValueError(
                    "Each key finding must be an object"
                )

            citations = finding.get("citations")

            if not isinstance(citations, list):
                raise ValueError(
                    "Each finding must contain a citations list"
                )

            if not all(
                isinstance(citation, str)
                and citation in allowed_citations
                for citation in citations
            ):
                raise ValueError(
                    "Report contains an invalid citation ID"
                )

        report["sources"] = evidence_for_report
        report["research_question"] = research_question
        report["rejected_evidence_count"] = len(
            rejected_evidence or []
        )

        return report


report_writer_agent = ReportWriterAgent()
