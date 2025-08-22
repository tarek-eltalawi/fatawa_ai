from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, BaseModel
from langchain_core.runnables import RunnableConfig
from typing import Any, List, cast
from src.retrieval_graph_with_tools.tools import TOOLS
from src.utilities.config import (TOOL_CALLING_MODEL, LOCAL_TOOL_CALLING_MODEL, LOCAL_REASONER_MODEL, OLLAMA_BASE_URL, 
    REASONER_MODEL, TEMPERATURE, QWQ_MODEL, OPENROUTER_API_KEY, OPENROUTER_API_BASE)


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str

# LLMs

tool_calling_llm = ChatOpenAI(
    temperature=TEMPERATURE,
    # model=TOOL_CALLING_MODEL,
    model="google/gemini-2.0-flash-exp:free",
    api_key=SecretStr(str(OPENROUTER_API_KEY)),
    base_url=OPENROUTER_API_BASE
)

reasoner_llm = ChatOpenAI(
    temperature=TEMPERATURE,
    model=REASONER_MODEL,
    api_key=SecretStr(str(OPENROUTER_API_KEY)),
    base_url=OPENROUTER_API_BASE
)

async def acall_generate_query(message: Any, config: RunnableConfig = None):
    """
    Generate a query from a question.

    Args:
        messages: The sequence of messages between user and system
        config, The RunnableConfig

    Returns:
        the generated query
    """
    model = tool_calling_llm.with_structured_output(SearchQuery)
    generated = cast(SearchQuery, await model.ainvoke(message, config))
    return generated.query

async def acall_reasoner(messages: Any, config: RunnableConfig = None):
    """
    Call the reasoner LLM with a list of messages.

    Args:
        messages: A list of messages to send to the LLM
        config: The RunnableConfig

    Returns:
        the generated response
    """
    return await reasoner_llm.ainvoke(messages, config)

async def acall_model_with_tools(messages: Any, config: RunnableConfig = None, tools: List[Any] = TOOLS):
    """
    Call the reasoner LLM with a list of messages.

    Args:
        messages: A list of messages to send to the LLM
        config: The RunnableConfig

    Returns:
        the generated response
    """
    bound_llm = tool_calling_llm.bind_tools(tools)
    return await bound_llm.ainvoke(messages, config)
