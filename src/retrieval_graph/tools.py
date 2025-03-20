"""
This module provides tools to be used by the RAG agent.
"""
from langchain_core.runnables import RunnableConfig
from src.retrieval_graph.retrieval import aretrieve_documents
from src.utilities.utils import sources_in_markdown
from langchain_core.runnables import RunnableConfig
from typing import Any, List, Callable

async def retrieve(query: str, language: str, config: RunnableConfig):
    """
    Fetches Islamic related documents from a vectorDB.
    Should be used for questions on Islamic jurisprudence, fiqh, Islamic law, any permissibility questions, and any questions related to the Quran and Sunnah.
    Args:
        query: The user's question.
        language: The language of the user's question.
        config: The configuration for this runnable.
    """
    result = await aretrieve_documents(query, config, language)
    return {
        "context": result["context"],
        "sources": sources_in_markdown(result["sources"], language == "ar")
    }

TOOLS: List[Callable[..., Any]] = [retrieve]
