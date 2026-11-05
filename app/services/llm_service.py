
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings


class LLMService:

    def __init__(self):

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.3
        )

    async def generate_response(self, prompt: str) -> str:

        response = await self.llm.ainvoke(prompt)

        content = response.content

        # Gemini can return text as a list of content blocks
        if isinstance(content, str):
            return content

        if isinstance(content, list):

            text_parts = []

            for block in content:

                if isinstance(block, str):
                    text_parts.append(block)

                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))

            return "\n".join(text_parts)

        raise ValueError("Gemini returned an unsupported response format")


llm_service = LLMService()
