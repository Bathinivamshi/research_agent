
import pytest

from app.rag.evidence import EvidenceVerifier


@pytest.fixture
def verifier():
    return EvidenceVerifier(minimum_similarity=0.55)


def test_supporting_evidence_is_found(verifier):

    claim = "Machine learning enables computers to learn from data."

    evidence = [
        {
            "text": (
                "Machine learning allows computer systems "
                "to identify patterns and learn from datasets."
            ),
            "document_name": "research.pdf",
            "page_number": 2
        }
    ]

    result = verifier.verify_claim(claim, evidence)

    assert result["verified"] is True
    assert len(result["evidence"]) == 1


def test_empty_evidence_is_not_verified(verifier):

    result = verifier.verify_claim(
        "Machine learning is useful.",
        []
    )

    assert result["verified"] is False
    assert result["reason"] == "No evidence provided"


def test_empty_claim_raises_error(verifier):

    with pytest.raises(ValueError):
        verifier.verify_claim("", [])


def test_unrelated_evidence_is_not_accepted(verifier):

    claim = "Machine learning enables computers to learn from data."

    evidence = [
        {
            "text": "The weather is sunny with clear blue skies.",
            "document_name": "weather.pdf",
            "page_number": 1
        }
    ]

    result = verifier.verify_claim(claim, evidence)

    assert result["verified"] is False


def test_invalid_threshold_raises_error():

    with pytest.raises(ValueError):
        EvidenceVerifier(minimum_similarity=1.5)
