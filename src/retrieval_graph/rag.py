from typing import Any, Dict
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval_graph.models import (language_detector, ainvoke, acall_generate_query)
from src.utilities.retrieval import aretrieve_documents
from src.utilities.state import State
from src.utilities.utils import sources_in_markdown
from src.utilities.prompts import (QUESTION_ROUTER_PROMPT, RESPONDER_PROMPT, QUERY_SYSTEM_PROMPT,
    RESPONSE_SYSTEM_PROMPT, SUMMARIZE_PROMPT)

async def generate_query(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.

    Args:
        state (State): The current state containing messages and other information.
        config (RunnableConfig | None, optional): Configuration for the query generation process.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing a list of generated queries.

    Behavior:
        - If there's only one message (first user input), it uses that as the query.
        - For subsequent messages, it uses a language model to generate a refined query.
        - The function uses the configuration to set up the prompt and model for query generation.
    """
    
    # It's the first user question. We will use the input directly to search.
    query = state.messages[-1].content
    config_with_nostream = {**config, "tags": ["langsmith:nostream"]}
    if len(state.messages) > 1:
        prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_SYSTEM_PROMPT),
            ("placeholder", "{messages}")])

        message_value = await prompt.ainvoke({"messages": state.messages, "queries": "\n- ".join(state.queries)},config_with_nostream)

        # Generate a query from the user's messages
        query = await acall_generate_query(message_value, config_with_nostream)
    
    return {
        "queries": [query]
    }

async def retrieval_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """
    Retrieve relevant documents based on the question from the state.
    
    Args:
        state: The current state containing question and language
        
    Returns:
        Dict containing context and sources to update the state
    """
    question = state.queries[-1]
    language = await language_detector.ainvoke({"question": question}, config)
    result = await aretrieve_documents(question, config, 'ar' in language.content.strip())
    
    return {
        "context": result["context"],
        "sources": sources_in_markdown(result["sources"], 'ar' in language.content.strip())
    }

async def model_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generate a response using the selected model and prompt.
    
    Args:
        state: The current state containing context and question
        
    Returns:
        Dict containing the generated response
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONSE_SYSTEM_PROMPT),
        ("placeholder", "{messages}")])

    message_value = await prompt.ainvoke({
        "messages": state.messages,
        "context": state.context,
        "sources": state.sources}, config)

    response = await ainvoke(message_value, config)
    return {
        "messages": [AIMessage(id=response.id, content=response.content)]
    }

async def respond_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Respond to the user's question."""
    question = state.queries[-1]
    system_prompt = RESPONDER_PROMPT
    response = await ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=question)], config)
    return {
        "messages": [AIMessage(id=response.id, content=response.content)]
    }

async def summarize_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Summarize the conversation."""
    summary_messages = [m for m in state.messages if getattr(m, 'name', None) == "summary"]
    existing_summary = summary_messages[0].content if summary_messages else ""
    filtered_messages = [
        msg for msg in state.messages 
        if (isinstance(msg, HumanMessage) or isinstance(msg, AIMessage)) and msg.content != ""
    ]
    
    template = SUMMARIZE_PROMPT
    summarize_prompt = PromptTemplate(template=template, input_variables=["summary", "messages"])
    prompt = summarize_prompt.format(messages=filtered_messages, summary=existing_summary)

    summary = await ainvoke(prompt, {**config, "tags": ["langsmith:nostream"]})
    summary_message = SystemMessage(name="summary", content=summary.content)
    # Delete all but the 2 most recent messages
    # delete_messages = [RemoveMessage(id=m.id) for m in filtered_messages[:-2] if m.id is not None]
    keep_ids = set()
    if len(filtered_messages) >= 2:
        keep_ids = {filtered_messages[-1].id, filtered_messages[-2].id}
    # Delete all messages except those with IDs to keep
    delete_messages = [RemoveMessage(id=m.id) for m in state.messages if m.id is not None and m.id not in keep_ids]

    return {
        "messages": [summary_message, *delete_messages]
    }

async def route_question(state: State) -> str:
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    template = QUESTION_ROUTER_PROMPT
    question_router_prompt = PromptTemplate(template=template, input_variables=["question"])
    prompt = question_router_prompt.format(question=state.queries[-1])
    source = await ainvoke(prompt, {"tags": ["langsmith:nostream"]})
    if "vectorstore" not in source.content:
        return "respond"
    else:
        return "retrieve"
    
async def should_summarize(state: State) -> str:
    """
    Determine if summarization is needed based on the state.

    Args:
        state (State): The current state of the conversation

    Returns:
        bool: True if summarization is needed, False otherwise
    """
    return "summarize" if len(state.messages) > 30 else "END"

graph_builder = StateGraph(State)
    
# Add nodes
graph_builder.add_node("generate_query", generate_query)
graph_builder.add_node("retrieve", retrieval_node)
graph_builder.add_node("generate_answer", model_node)
graph_builder.add_node("respond", respond_node)
graph_builder.add_node("summarize", summarize_node)

# Add edges
graph_builder.add_edge(START, "generate_query")
graph_builder.add_conditional_edges(
    "generate_query",
    route_question,
    {
        "respond": "respond",
        "retrieve": "retrieve",
    },
)
graph_builder.add_edge("retrieve", "generate_answer")
graph_builder.add_conditional_edges(
    "generate_answer",
    should_summarize,
    {
        "summarize": "summarize",
        "END": END,
    },
)
graph_builder.add_conditional_edges(
    "respond",
    should_summarize,
    {
        "summarize": "summarize",
        "END": END,
    },
)
graph_builder.add_edge("summarize", END)
graph = graph_builder.compile()