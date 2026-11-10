
import pytest

from app.agents.report_writer import ReportWriterAgent


class MockLLM:
    async def generate_response(self, prompt):
        return """
        {
            "title": "AI in Business",
            "executive_summary": "AI can automate repetitive tasks.",
            "introduction": "This report examines AI in business.",
            "key_findings": [
                {
                    "finding": "AI can automate repetitive tasks.",
                    "citations": ["S1"]
                }
            ],
            "analysis": "AI supports business automation [S1].",
            "limitations": [
                "The report uses a limited number of sources."
            ],
            "conclusion": "AI can support business productivity [S1]."
        }
        """


@pytest.fixture
def writer():
    return ReportWriterAgent(llm=MockLLM())


@pytest.fixture
def evidence():
    return [
        {
            "claim": "AI can automate repetitive tasks.",
            "supporting_quote": "AI can automate repetitive tasks.",
            "source_title": "AI Research",
            "source_url": "https://example.com/ai",
            "verification_reason": "The quote supports the claim."
        }
    ]


@pytest.mark.asyncio
async def test_generate_report(writer, evidence):
    report = await writer.generate_report(
        research_question="How does AI help businesses?",
        verified_evidence=evidence
    )

    assert report["title"] == "AI in Business"
    assert len(report["sources"]) == 1
    assert report["sources"][0]["citation_id"] == "S1"


@pytest.mark.asyncio
async def test_empty_question_raises_error(writer, evidence):
    with pytest.raises(ValueError):
        await writer.generate_report("", evidence)


@pytest.mark.asyncio
async def test_empty_evidence_raises_error(writer):
    with pytest.raises(ValueError):
        await writer.generate_report("What is AI?", [])


@pytest.mark.asyncio
async def test_invalid_citation_is_rejected(writer, evidence):
    class InvalidCitationLLM:
        async def generate_response(self, prompt):
            return """
            {
                "title": "AI Report",
                "executive_summary": "Summary",
                "introduction": "Introduction",
                "key_findings": [
                    {
                        "finding": "AI finding",
                        "citations": ["S99"]
                    }
                ],
                "analysis": "Analysis",
                "limitations": [],
                "conclusion": "Conclusion"
            }
            """

    invalid_writer = ReportWriterAgent(llm=InvalidCitationLLM())

    with pytest.raises(ValueError):
        await invalid_writer.generate_report(
            "What is AI?",
            evidence
        )
