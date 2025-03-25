from mcp.server.fastmcp import FastMCP
from langchain_core.runnables import RunnableConfig
from typing import Dict
from typing import Any
from src.utilities.retrieval import aretrieve_documents
from src.utilities.utils import sources_in_markdown

mcp = FastMCP("DarAlIftaa")

@mcp.tool()
async def retrieve_islamic_docs(question: str, is_arabic: bool, config: RunnableConfig = None
) -> Dict[str, Any]:
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
    result = await aretrieve_documents(question, config, is_arabic)
    return {
        "context": result["context"] + sources_in_markdown(result["sources"], is_arabic)
    }

if __name__ == "__main__":
    mcp.run(transport="sse")