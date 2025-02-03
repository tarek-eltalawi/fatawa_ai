from langchain.agents import Tool, AgentExecutor
from tools.dar_al_iftaa_qa_tool import DarAlIftaQATool, qa_tool_func
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.agents import ZeroShotAgent
from config import (
    OLLAMA_BASE_URL,
    MODEL_NAME,
    TEMPERATURE,
    AGENT_TEMPLATE
)

class CustomIslamicAgent(ZeroShotAgent):
    @classmethod
    def create_prompt(cls, tools, **kwargs):
        # Create a prompt that only requires input and agent_scratchpad
        return PromptTemplate(
            template=AGENT_TEMPLATE,
            input_variables=["input", "agent_scratchpad"]
        )

def create_islamic_agent():
    # Instantiate base LLM for the agent
    llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=MODEL_NAME, temperature=TEMPERATURE)

    # Define tools
    tools = [
        Tool(
            name=DarAlIftaQATool.get_instance().name,
            func=qa_tool_func,
            description=DarAlIftaQATool.get_instance().description
        )
    ]
    
    agent = CustomIslamicAgent.from_llm_and_tools(llm=llm, tools=tools)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    # Initialize and return the agent
    # return initialize_agent(
    #     tools,
    #     llm,
    #     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    #     handle_parsing_errors=True,
    #     agent_kwargs={
    #         "prefix": AGENT_PREFIX,
    #         "suffix": AGENT_SUFFIX
    #     }
    # )

    # Create the agent using react, since it works better when you have 1 tool
    # agent = create_react_agent(
    #     llm=llm,
    #     tools=tools,
    #     prompt=prompt
    # )

    # # Create the agent executor
    # return AgentExecutor(
    #     agent=agent,
    #     tools=tools,
    #     verbose=True,
    #     handle_parsing_errors=True
    # )