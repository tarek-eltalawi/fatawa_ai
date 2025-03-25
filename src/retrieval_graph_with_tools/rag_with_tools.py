from typing import Any, Dict, Literal
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from src.retrieval_graph_with_tools.models import (acall_generate_query, acall_model_with_tools, acall_reasoner)
from src.retrieval_graph_with_tools.tools import TOOLS
from src.utilities.state import State
from src.utilities.prompts import (QUERY_SYSTEM_PROMPT, RESPONSE_SYSTEM_PROMPT_WITH_TOOLS, SUMMARIZE_PROMPT)

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
    # If there is summary, then we add it
    if state.summary:    
        messages = [SystemMessage(content=state.summary), *state.messages]

    # It's the first user question. We will use the input directly to search.
    query = messages[-1].content
    if len(messages) > 1:
        prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_SYSTEM_PROMPT),
            ("placeholder", "{messages}")])

        message_value = await prompt.ainvoke({"messages": messages, "queries": "\n- ".join(state.queries)},config)

        # Generate a query from the user's messages
        query = await acall_generate_query(message_value, config)
    
    return {
        "queries": [query]
    }

async def answer(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generate a response using the selected model and prompt.
    
    Args:
        state: The current state containing context and question
        
    Returns:
        Dict containing the generated response
    """

    messages = state.messages
    last_message = state.messages[-1]

    # If there is summary, then we add it
    if state.summary:    
        messages = [SystemMessage(content=state.summary), *state.messages]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONSE_SYSTEM_PROMPT_WITH_TOOLS),
        ("placeholder", "{messages}")])

    message_value = await prompt.ainvoke({"messages": messages}, config)
    # TODO: implement a step to prevent loops
    if last_message and last_message.type == "tool" and last_message.name == "retrieve":
        response = await acall_reasoner(message_value, config)
    else:
        response = await acall_model_with_tools(message_value, config)
    return {
        "messages": [response]
    }

async def summarize(state: State, *, config: RunnableConfig) -> Dict[str, Any]:
    """Summarize the conversation."""
    filtered_messages = [
        msg for msg in state.messages 
        if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage)
    ]
    template = SUMMARIZE_PROMPT
    summarize_prompt = PromptTemplate(template=template, input_variables=["summary", "messages"])
    prompt = summarize_prompt.format(messages=filtered_messages, summary=state.summary)
    summary = await acall_reasoner(prompt, config)
    # Delete all but the 2 most recent messages
    # Get the IDs of the last human and AI messages from filtered_messages
    keep_ids = set()
    if len(filtered_messages) >= 2:
        keep_ids = {filtered_messages[-1].id, filtered_messages[-2].id}
    # Delete all messages except those with IDs to keep
    delete_messages = [
        RemoveMessage(id=m.id) for m in state.messages 
        if m.id is not None and m.id not in keep_ids
    ]

    return {
        "summary": summary.content,
        "messages": delete_messages
    }

def should_use_tools_or_summarize_or_end(state: State) -> Literal["END", "tools", "summarize"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.
    If no tools to call then it checks if a summarization should be done.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("END" or "tools" or "summarize").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )

    if last_message.tool_calls:        
        return "tools"
    return "summarize" if len(state.messages) > 30 else "END"

memory = MemorySaver()
graph_builder = StateGraph(State)
    
# Add nodes
graph_builder.add_node("generate_query", generate_query)
graph_builder.add_node("answer", answer)
graph_builder.add_node("summarize", summarize)
graph_builder.add_node("tools", ToolNode(TOOLS))

# Add edges
graph_builder.add_edge(START, "generate_query")
graph_builder.add_edge("generate_query", "answer")
graph_builder.add_conditional_edges(
    "answer",
    should_use_tools_or_summarize_or_end,
    {
        "summarize": "summarize",
        "tools": "tools",
        "END": END,
    },
)
graph_builder.add_edge("tools", "answer")
graph_builder.add_edge("summarize", END)
graph = graph_builder.compile(checkpointer=memory)