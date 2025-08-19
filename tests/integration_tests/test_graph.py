import uuid
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from src.retrieval_graph import graph
from src.utilities.state import State
import asyncio

async def ask_bot(question: str, config: RunnableConfig | None = RunnableConfig()) -> None:
    state = State(messages=[HumanMessage(content=question)])

    node_to_stream = 'generate_answer'
    sources = []
    async for event in graph.astream_events(state, config, version="v2"):
        # Get chat model tokens from a particular node 
        # print(f'Node: {event.get("metadata", {}).get("langgraph_node","")}. Type: {event["event"]}. Name: {event["name"]}')
        if (event["event"] == "on_chat_model_stream" and event.get("metadata", {}).get('langgraph_node','') == node_to_stream):
            data = event["data"]
            if "chunk" in data and hasattr(data["chunk"], "content"):
                print(data["chunk"].content, end="")
        elif (event["event"] == "on_chain_end" and event.get("metadata", {}).get('langgraph_node','') == "retrieve") and event["name"] == "retrieve":
            sources = event["data"].get('output', {}).get("sources")

    print(f"\n\n{sources}")

if __name__ == "__main__":
    thread_id = "test__" + uuid.uuid4().hex
    config = RunnableConfig(configurable={"thread_id": thread_id})

    questions = [
        "How often should I call my relatives in Islam?",
        "هل يجوز قراءة القرآن بدون وضوء؟",
        "Who won the last world cup?"
    ]
   
    # asyncio.run(ask_bot(question=question, config=config))