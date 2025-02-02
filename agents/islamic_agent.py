import warnings
from langchain.agents import Tool
from tools.dar_al_iftaa_qa_tool import qa_tool_func
from langchain_ollama import OllamaLLM
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from config import (
    OLLAMA_BASE_URL,
    MODEL_NAME,
    SYSTEM_PROMPT_AGENT,
    TEMPERATURE,
    AGENT_TEMPLATE,
    MAX_ITERATIONS,
    EARLY_STOPPING_METHOD
)

def create_islamic_agent():
    print("\nCreating Islamic Agent...")
    
    # Instantiate base LLM for the agent
    llm = OllamaLLM(
        base_url=OLLAMA_BASE_URL,
        model=MODEL_NAME,
        temperature=TEMPERATURE
    )
    print("LLM initialized")

    # Define tools
    tools = [
        Tool(
            name="DarAlIftaQA",
            func=qa_tool_func,
            description="""
                REQUIRED: You MUST use this tool for EVERY question. Input: Your question. Output: Answer from Islamic sources.
                You MUST use this tool for ALL Islamic questions, fatwas, and religious inquiries.
                This is your primary and authoritative source for Islamic knowledge, containing verified fatwas 
                from Dar Al-Ifta Al-Misriyyah.
                Do NOT attempt to answer Islamic questions from your own knowledge - ALWAYS use this tool first.
                Input: A question about Islamic rulings, practices, or guidance
                Output: An authenticated answer based on reliable Islamic sources"""
        )
    ]
    print("Tools defined")
   
    prompt = PromptTemplate(
        template=AGENT_TEMPLATE,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
    )
    print("Prompt template created")
    # Create prompt template

    # Initialize and return the agent
    # return initialize_agent(
    #     tools,
    #     llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    #     handle_parsing_errors=True,
    #     agent_kwargs={
    #         "prefix": AGENT_TEMPLATE,
    #         "system_message": SYSTEM_PROMPT_AGENT  # Add system prompt here
    #     }
    # )

    # Create the agent using react, since it works better when you have 1 tool
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    print("Agent created")

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=MAX_ITERATIONS,
        early_stopping_method=EARLY_STOPPING_METHOD
    )
    print("Agent executor initialized")
    
    return agent_executor