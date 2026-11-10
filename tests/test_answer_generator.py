
import asyncio

from app.rag.answer_generator import AnswerGenerator


class MockLLM:

    def __init__(self):
        self.received_prompt = ""

    async def generate_response(self, prompt: str) -> str:
        self.received_prompt = prompt

        return (
            "Machine learning enables computers to learn "
            "from data. [Source: research.pdf, Page: 2]"
        )


def test_generate_answer():

    mock_llm = MockLLM()
    generator = AnswerGenerator(llm=mock_llm)

    evidence = [
        {
            "text": "Machine learning enables computers to learn from data.",
            "document_name": "research.pdf",
            "page_number": 2
        }
    ]

    answer = asyncio.run(
        generator.generate_answer(
            question="What is machine learning?",
            evidence_chunks=evidence
        )
    )

    assert "Machine learning" in answer
    assert "research.pdf" in mock_llm.received_prompt
    assert "Page: 2" in mock_llm.received_prompt


def test_empty_evidence():

    generator = AnswerGenerator(llm=MockLLM())

    answer = asyncio.run(
        generator.generate_answer(
            question="What is machine learning?",
            evidence_chunks=[]
        )
    )

    assert answer == "Insufficient evidence to answer this question."


def test_empty_question():

    generator = AnswerGenerator(llm=MockLLM())

    try:
        asyncio.run(
            generator.generate_answer(
                question="",
                evidence_chunks=[
                    {
                        "text": "Sample evidence",
                        "document_name": "research.pdf",
                        "page_number": 1
                    }
                ]
            )
        )

        assert False, "Expected ValueError"

    except ValueError:
        pass
