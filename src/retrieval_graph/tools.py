"""
This module provides tools to be used by the RAG agent.
"""
from langchain_core.runnables import RunnableConfig
from src.retrieval_graph.retrieval import aretrieve_documents
from src.utilities.utils import sources_in_markdown
from langchain_core.runnables import RunnableConfig
from typing import Any, List, Callable

async def retrieve(query: str, is_arabic: bool, config: RunnableConfig):
    """
    Fetches Islamic related documents from a vectorDB.
    Should be used for questions on Islamic jurisprudence, fiqh, Islamic law, any permissibility questions, and any questions related to the Quran and Sunnah.
    Args:
        query: The user's question.
        is_arabic: A boolean indicating if the user's question is in Arabic.
        config: The configuration for this runnable.
    Returns:
        Dict containing context and sources in markdown format.
    """
    result = await aretrieve_documents(query, config, is_arabic)
    return {
        "context": result["context"],
        "sources": sources_in_markdown(result["sources"], is_arabic)
    }

TOOLS: List[Callable[..., Any]] = [retrieve]
