
import pytest

from app.agents.evidence_extractor import EvidenceExtractorAgent


class MockLLM:
    async def generate_response(self, prompt):
        return """
        {
            "evidence": [
                {
                    "claim": "AI can automate repetitive tasks.",
                    "supporting_quote": "Artificial intelligence can automate repetitive tasks."
                },
                {
                    "claim": "Unsupported claim.",
                    "supporting_quote": "This sentence does not exist in the source."
                }
            ]
        }
        """


@pytest.fixture
def extractor():
    return EvidenceExtractorAgent(llm=MockLLM())


@pytest.mark.asyncio
async def test_extract_evidence_from_document(extractor):
    document = {
        "title": "AI Research",
        "url": "https://example.com/ai",
        "content": (
            "Artificial intelligence can automate repetitive tasks. "
            "It is used in many industries."
        )
    }

    result = await extractor.extract_from_document(
        research_question="How does AI help businesses?",
        document=document
    )

    assert len(result) == 1
    assert result[0]["evidence_verified"] is True
    assert result[0]["source_title"] == "AI Research"


def test_validate_quote(extractor):
    content = "AI improves productivity in businesses."

    assert extractor.validate_quote(
        "AI improves productivity in businesses.",
        content
    )

    assert not extractor.validate_quote(
        "AI replaces every employee.",
        content
    )


@pytest.mark.asyncio
async def test_empty_question_raises_error(extractor):
    with pytest.raises(ValueError):
        await extractor.extract_evidence("", [{"content": "Some text"}])


@pytest.mark.asyncio
async def test_empty_documents_raise_error(extractor):
    with pytest.raises(ValueError):
        await extractor.extract_evidence("What is AI?", [])
