
import json
import re

from app.services.llm_service import llm_service


class ResearchPlanner:
    def __init__(self, llm=None):
        self.llm = llm or llm_service

    def parse_json_response(self, response: str) -> dict:
        """Extract JSON even when Gemini adds Markdown or extra text."""

        if not isinstance(response, str) or not response.strip():
            raise ValueError("Research planner returned an empty response")

        cleaned = response.strip()

        # Remove Markdown JSON code fences.
        cleaned = re.sub(
            r"```(?:json)?",
            "",
            cleaned,
            flags=re.IGNORECASE
        ).replace("```", "").strip()

        # Find the first JSON object in the response.
        decoder = json.JSONDecoder()

        start = cleaned.find("{")

        while start != -1:
            try:
                parsed, _ = decoder.raw_decode(cleaned[start:])

                if isinstance(parsed, dict):
                    return parsed

            except json.JSONDecodeError:
                start = cleaned.find("{", start + 1)
                continue

        raise ValueError(
            "Research planner returned invalid JSON. "
            f"Gemini response preview: {response[:500]}"
        )

    async def create_plan(self, research_question: str) -> dict:

        if not research_question or not research_question.strip():
            raise ValueError("Research question cannot be empty")

        prompt = f"""
You are an expert research planning assistant.

Research question:
{research_question}

Create a research plan.

Requirements:
1. Identify the main research topic.
2. Create 3 to 5 focused research questions.
3. Create 3 to 5 specific web search queries.
4. Cover different aspects of the research topic.
5. Avoid duplicate search queries.
6. Make queries suitable for web search.

Return ONLY a valid JSON object.
Do not use Markdown.
Do not add explanations before or after the JSON.

Required structure:

{{
    "topic": "Main research topic",
    "research_questions": [
        "Research question 1",
        "Research question 2",
        "Research question 3"
    ],
    "search_queries": [
        "Search query 1",
        "Search query 2",
        "Search query 3"
    ]
}}
"""

        response = await self.llm.generate_response(prompt)

        plan = self.parse_json_response(response)

        topic = plan.get("topic")
        research_questions = plan.get("research_questions")
        search_queries = plan.get("search_queries")

        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("Research planner returned an invalid topic")

        if not isinstance(research_questions, list):
            raise ValueError("Research questions must be a list")

        if not isinstance(search_queries, list):
            raise ValueError("Search queries must be a list")

        if not 3 <= len(research_questions) <= 5:
            raise ValueError("Planner must generate 3 to 5 research questions")

        if not 3 <= len(search_queries) <= 5:
            raise ValueError("Planner must generate 3 to 5 search queries")

        if not all(
            isinstance(item, str) and item.strip()
            for item in research_questions
        ):
            raise ValueError("Planner returned invalid research questions")

        if not all(
            isinstance(item, str) and item.strip()
            for item in search_queries
        ):
            raise ValueError("Planner returned invalid search queries")

        return {
            "topic": topic.strip(),
            "research_questions": research_questions,
            "search_queries": search_queries
        }


research_planner = ResearchPlanner()
