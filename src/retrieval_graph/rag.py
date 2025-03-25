from typing import Any, Dict
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval_graph.models import (language_detector, acall_reasoner, query_generator)
from src.utilities.retrieval import aretrieve_documents
from src.utilities.state import State
from src.utilities.utils import sources_in_markdown
from src.utilities.prompts import (QUESTION_ROUTER_PROMPT, RESPONDER_PROMPT, 
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
    
    messages = state.messages
    question = messages[-1].content
    # If there is summary, then we add it
    if state.summary:    
        messages = [SystemMessage(content=state.summary), *state.messages]

    response = await query_generator.ainvoke({
        "queries": "\n- ".join(state.queries),
        "question": question,
        "messages": messages}, config)
    
    return {
        "queries": [response.content]
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
    result = await aretrieve_documents(question, config, 'ar' in language.content)
    
    return {
        "context": result["context"],
        "sources": sources_in_markdown(result["sources"], 'ar' in language.content)
    }

async def model_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generate a response using the selected model and prompt.
    
    Args:
        state: The current state containing context and question
        
    Returns:
        Dict containing the generated response
    """

    messages = state.messages
    # If there is summary, then we add it
    if state.summary:    
        messages = [SystemMessage(content=state.summary), *state.messages]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONSE_SYSTEM_PROMPT),
        ("placeholder", "{messages}")])

    message_value = await prompt.ainvoke({
        "messages": messages,
        "context": state.context,
        "sources": state.sources}, config)

    response = await acall_reasoner(message_value, config)
    return {
        "messages": [AIMessage(id=response.id, content=response.content)]
    }

async def respond_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Respond to the user's question."""
    question = state.queries[-1]
    system_prompt = RESPONDER_PROMPT
    response = await acall_reasoner([SystemMessage(content=system_prompt), HumanMessage(content=question)], config)
    return {
        "messages": [AIMessage(id=response.id, content=response.content)]
    }

async def summarize_node(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Summarize the conversation."""
    template = SUMMARIZE_PROMPT
    summarize_prompt = PromptTemplate(template=template, input_variables=["summary", "messages"])
    prompt = summarize_prompt.format(messages=state.messages, summary=state.summary)
    summary = await acall_reasoner(prompt, config)
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state.messages[:-2] if m.id is not None]

    return {
        "summary": summary.content,
        "messages": delete_messages
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
    source = await acall_reasoner(prompt)
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
    return "summarize" if len(state.messages) > 3 else "END"

memory = MemorySaver()
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
graph = graph_builder.compile(checkpointer=memory)