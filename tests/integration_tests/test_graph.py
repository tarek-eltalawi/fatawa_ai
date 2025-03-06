import uuid
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from src.retrieval_graph import graph
from src.retrieval_graph.state import State
import asyncio

async def ask_bot(question: str, lang: str = "en", config: RunnableConfig | None = RunnableConfig()) -> None:
    summary = graph.get_state(config or RunnableConfig()).values.get('summary', "")
    state = State(messages=[HumanMessage(content=question)], language=lang, summary=summary)

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
    # question = "can i greet christians?"
    # asyncio.run(ask_bot(question=question, config=config))
    # question = "can i listen to music?"
    # asyncio.run(ask_bot(question=question, config=config))
    # question = "what about jews?"
    # asyncio.run(ask_bot(question=question, config=config))
    # question = "what about buddhists?"
    # asyncio.run(ask_bot(question=question, config=config))

    question = "هل يجوز قراءة القرآن بدون وضوء؟"
    asyncio.run(ask_bot(question=question, config=config, lang="ar"))
    # question = "ما حكم إخراج زكاة المال في شكل إفطارٍ للصائمين؟"
    # asyncio.run(ask_bot(question=question, config=config, lang="ar"))
    # question = "ممكن أسمع موسيقى ؟"
    # asyncio.run(ask_bot(question=question, config=config, lang="ar"))