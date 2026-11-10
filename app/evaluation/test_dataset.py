# TODO: Maintain a repeatable research evaluation dataset.

class ResearchTestDataset:

    def __init__(self):
        self.dataset = [
            {
                "id": "test_001",
                "category": "RAG",
                "question": "What is Retrieval-Augmented Generation?",
                "expected_key_points": [
                    "Retrieves relevant external information",
                    "Provides retrieved context to a language model",
                    "Generates answers grounded in retrieved information"
                ],
                "relevant_document_ids": [
                    "doc_001",
                    "doc_002"
                ],
                "expected_sources": [
                    "source_001",
                    "source_002"
                ]
            },
            {
                "id": "test_002",
                "category": "Agentic AI",
                "question": "What is an AI agent?",
                "expected_key_points": [
                    "An AI agent works toward a defined goal",
                    "An agent can use tools",
                    "An agent can perform multiple reasoning or action steps"
                ],
                "relevant_document_ids": [
                    "doc_003",
                    "doc_004"
                ],
                "expected_sources": [
                    "source_003"
                ]
            },
            {
                "id": "test_003",
                "category": "LLM Evaluation",
                "question": "What is faithfulness in LLM evaluation?",
                "expected_key_points": [
                    "Measures whether generated claims are supported by context",
                    "Detects unsupported statements",
                    "Evaluates grounding in supplied evidence"
                ],
                "relevant_document_ids": [
                    "doc_005",
                    "doc_006"
                ],
                "expected_sources": [
                    "source_004"
                ]
            },
            {
                "id": "test_004",
                "category": "Embeddings",
                "question": "What are text embeddings?",
                "expected_key_points": [
                    "Represent text as numerical vectors",
                    "Capture semantic information",
                    "Support similarity-based retrieval"
                ],
                "relevant_document_ids": [
                    "doc_007",
                    "doc_008"
                ],
                "expected_sources": [
                    "source_005"
                ]
            },
            {
                "id": "test_005",
                "category": "Vector Databases",
                "question": "Why are vector databases used in RAG systems?",
                "expected_key_points": [
                    "Store vector representations",
                    "Enable similarity search",
                    "Retrieve relevant context for language models"
                ],
                "relevant_document_ids": [
                    "doc_009",
                    "doc_010"
                ],
                "expected_sources": [
                    "source_006"
                ]
            }
        ]

    def get_all_cases(self) -> list:
        """Return all evaluation cases."""
        return self.dataset.copy()

    def get_case_by_id(self, case_id: str) -> dict | None:
        """Find a test case using its ID."""

        for case in self.dataset:
            if case["id"] == case_id:
                return case.copy()

        return None

    def get_cases_by_category(self, category: str) -> list:
        """Return cases belonging to a particular category."""

        return [
            case.copy()
            for case in self.dataset
            if case["category"].lower() == category.lower()
        ]

    def get_dataset_summary(self) -> dict:
        """Return dataset statistics."""

        categories = {}

        for case in self.dataset:
            category = case["category"]
            categories[category] = categories.get(category, 0) + 1

        return {
            "total_cases": len(self.dataset),
            "categories": categories
        }

    def validate_dataset(self) -> bool:
        """Validate required fields and unique test IDs."""

        required_fields = {
            "id",
            "category",
            "question",
            "expected_key_points",
            "relevant_document_ids",
            "expected_sources"
        }

        seen_ids = set()

        for case in self.dataset:

            if not required_fields.issubset(case.keys()):
                return False

            if case["id"] in seen_ids:
                return False

            seen_ids.add(case["id"])

            if not isinstance(case["question"], str) or not case["question"].strip():
                return False

            if not isinstance(case["expected_key_points"], list):
                return False

            if not isinstance(case["relevant_document_ids"], list):
                return False

        return True


research_test_dataset = ResearchTestDataset()
