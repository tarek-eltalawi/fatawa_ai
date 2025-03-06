from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from src.retrieval_graph.config import (INTERACTIVE_MODEL, LOCAL_INTERACTIVE_MODEL, LOCAL_REASONER_MODEL, OLLAMA_BASE_URL, 
    REASONER_MODEL, TEMPERATURE, QWQ_MODEL, OPENROUTER_API_KEY, OPENROUTER_API_BASE, OPENROUTER_QWQ_MODEL)
from src.retrieval_graph.prompts import (
    RESPONSE_SYSTEM_PROMPT_EN,
    RESPONSE_SYSTEM_PROMPT_AR
)
from openai import OpenAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.runnables import RunnableConfig
from typing import Any
from pydantic import BaseModel
from typing import cast

class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str

# LLMs
interactive_llm = ChatOpenAI(
    temperature=TEMPERATURE,
    model=INTERACTIVE_MODEL,
    api_key=SecretStr(str(OPENROUTER_API_KEY)),
    base_url=OPENROUTER_API_BASE
)

reasoner_llm = ChatOpenAI(
    temperature=TEMPERATURE,
    model=REASONER_MODEL,
    api_key=SecretStr(str(OPENROUTER_API_KEY)),
    base_url=OPENROUTER_API_BASE
)

local_llm = ChatOllama(model=QWQ_MODEL, temperature=TEMPERATURE, base_url=OLLAMA_BASE_URL)

async def acall_generate_query(message: Any, config: RunnableConfig):
    """
    Generate a query from a question.

    Args:
        messages: The sequence of messages between user and system
        config, The RunnableConfig

    Returns:
        the generated query
    """
    llm = interactive_llm if LOCAL_INTERACTIVE_MODEL is "" else local_llm
    model = llm.with_structured_output(SearchQuery)
    generated = cast(SearchQuery, await model.ainvoke(message, config))
    return generated.query

async def acall_interactive(messages: Any, config: RunnableConfig = RunnableConfig()):
    """
    Call the reasoner LLM with a list of messages.

    Args:
        messages: A list of messages to send to the LLM
        config: The RunnableConfig

    Returns:
        the generated response
    """
    llm = interactive_llm if LOCAL_INTERACTIVE_MODEL is "" else local_llm
    return await llm.ainvoke(messages, config)

async def acall_reasoner(messages: Any, config: RunnableConfig):
    """
    Call the reasoner LLM with a list of messages.

    Args:
        messages: A list of messages to send to the LLM
        config: The RunnableConfig

    Returns:
        the generated response
    """
    llm = reasoner_llm if LOCAL_REASONER_MODEL is "" else local_llm
    return await llm.ainvoke(messages, config)


###
# Below methods are not used, but will be in the future
###

def call_openrouter(state, messages):
    client = OpenAI(
    base_url=OPENROUTER_API_BASE,
    api_key=OPENROUTER_API_KEY,
    )
    openai_messages = []
    # Add system message
    system_content = RESPONSE_SYSTEM_PROMPT_AR if state.language == "ar" else RESPONSE_SYSTEM_PROMPT_EN
    system_content = system_content.format(context=state.context, messages=messages)
    openai_messages.append({"role": "system", "content": system_content})
    
    # Add user messages
    for msg in messages:
        if msg.type == "human":
            openai_messages.append({"role": "user", "content": msg.content})
        elif msg.type == "ai":
            openai_messages.append({"role": "assistant", "content": msg.content})

    completion = client.chat.completions.create(
        model=OPENROUTER_QWQ_MODEL,
        messages=openai_messages
    )
    response = completion.choices[0].message.content
    # reasoning = completion.choices[0].reasoning

#########
# LLMs
llm_json_format = ChatOllama(model=QWQ_MODEL, format="json", temperature=TEMPERATURE, base_url=OLLAMA_BASE_URL)

### Retrieval Grader

retrieval_grader_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)

retrieval_grader = retrieval_grader_prompt | llm_json_format | JsonOutputParser()

### Generate

# Prompt
rag_prompt = hub.pull("rlm/rag-prompt")

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = rag_prompt | local_llm | StrOutputParser()

### Question Re-writer

# Prompt
re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
    input_variables=["generation", "question"],
)

question_rewriter = re_write_prompt | local_llm | StrOutputParser()

### Answer Grader

# Prompt
answer_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)

answer_grader = answer_grader_prompt | llm_json_format | JsonOutputParser()

### Hallucination Grader

# Prompt
hallucination_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)

hallucination_grader = hallucination_grader_prompt | llm_json_format | JsonOutputParser()