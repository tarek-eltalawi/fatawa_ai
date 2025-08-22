from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from typing import Any, cast
from src.utilities.config import TEMPERATURE

class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str

llm = ChatOpenAI(
    temperature=TEMPERATURE,
    model="gpt-4o-mini",
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
    model = llm.with_structured_output(SearchQuery)
    generated = cast(SearchQuery, await model.ainvoke(message, config))
    return generated.query

async def ainvoke(messages: Any, config: RunnableConfig = None):
    """
    Call the LLM with a list of messages.

    Args:
        messages: A list of messages to send to the LLM
        config: The RunnableConfig

    Returns:
        the generated response
    """
    return await llm.ainvoke(messages, config)

### Language detector

language_detector_prompt = PromptTemplate(
    template="""
    Detect the language of the question and return its universal code.
    e.g: for Arabic return 'ar' for English return `en`

    question: {question}
    """,
    input_variables=["question"],
)

language_detector = language_detector_prompt | llm.with_config({ "tags": ["langsmith:nostream"] })
