from contextlib import asynccontextmanager
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from src.retrieval_graph_with_mcp.models import tool_calling_llm
from src.utilities.prompts import RESPONSE_SYSTEM_PROMPT_WITH_TOOLS

@asynccontextmanager
async def graph():
    async with MultiServerMCPClient(
        {
            "DarAlIftaa": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    ) as client:
        agent = create_react_agent(
            model=tool_calling_llm,
            tools=client.get_tools(),
            prompt=RESPONSE_SYSTEM_PROMPT_WITH_TOOLS,
            checkpointer=MemorySaver()
        )
        yield agent

