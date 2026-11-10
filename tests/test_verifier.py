
import pytest

from app.agents.verifier import VerifierAgent


class MockLLM:
    async def generate_response(self, prompt):
        return """
        {
            "results": [
                {
                    "id": 0,
                    "supported": true,
                    "reason": "The quote directly supports the claim."
                }
            ]
        }
        """


@pytest.fixture
def verifier():
    return VerifierAgent(llm=MockLLM())


@pytest.mark.asyncio
async def test_verify_supported_claim(verifier):
    evidence = [
        {
            "claim": "AI can automate repetitive tasks.",
            "supporting_quote": "AI can automate repetitive tasks.",
            "source_content": "AI can automate repetitive tasks.",
            "source_title": "AI Research"
        }
    ]

    result = await verifier.verify_claims(
        "How does AI help businesses?",
        evidence
    )

    assert result["verified_claims"] == 1
    assert result["verification_passed"] is True


@pytest.mark.asyncio
async def test_reject_quote_not_found(verifier):
    evidence = [
        {
            "claim": "AI replaces every employee.",
            "supporting_quote": "AI replaces every employee.",
            "source_content": "AI helps automate repetitive tasks."
        }
    ]

    result = await verifier.verify_claims("What is AI?", evidence)

    assert result["verified_claims"] == 0
    assert result["rejected_claims"] == 1


@pytest.mark.asyncio
async def test_empty_evidence(verifier):
    result = await verifier.verify_claims("What is AI?", [])

    assert result["verification_passed"] is False
    assert result["verified_claims"] == 0


@pytest.mark.asyncio
async def test_empty_question(verifier):
    with pytest.raises(ValueError):
        await verifier.verify_claims("", [])
