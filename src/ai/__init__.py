"""
AI Advisor package (v5.9).

Uses the user's CodeCraft API key (https://codecraftapi.com) - an
OpenAI-compatible LLM relay - to add a short Arabic technical explanation
("تحليل ذكي") to each recommendation. Optional layer: the bot works
exactly as before if the key is missing or the service is unreachable.
"""
from src.ai.llm_advisor import ai_advisor, LLMAdvisor

__all__ = ["ai_advisor", "LLMAdvisor"]
