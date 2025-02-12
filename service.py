from typing import Annotated, Any, Dict, List, Literal, Sequence
from dataclasses import dataclass, field
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from config import JAIS_MODEL, QWEN_MODEL, OLLAMA_BASE_URL, TEMPERATURE, TOP_K
from pinecone_manager import PineconeManager
from tools import TOOLS, detect_language
from memory import ConversationMemory
from prompts import JAIS_PROMPT, QWEN_PROMPT

# Initialize conversation memory
memory = ConversationMemory(max_messages=10)

@dataclass
class State:
    question: str = field(default_factory=str)
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    context: str = field(default_factory=str)
    sources: List[str] = field(default_factory=list)
    response: str = field(default_factory=str)
    history: str = field(default_factory=str)
    language: str = field(default_factory=str)

# Retrieval node: adds a "context" key based on the question.
def retrieval_node(state: State) -> Dict[str, Any]:
    namespace = "Default" if state.language == 'ar' else "my_documents"
    pinecone_manager = PineconeManager(namespace=namespace)
    question = state.question
    retriever = pinecone_manager.vector_store.as_retriever(search_kwargs={"k": TOP_K})
    retrieved_docs = retriever.invoke(question)
    # Combine the content of retrieved docs into a single string.
    context = "\n".join(doc.page_content for doc in retrieved_docs)
    sources = set(
        doc.metadata.get('source', 'No source available') if doc.metadata.get('source', '').startswith('https://www.dar-alifta.org')
        else "https://www.dar-alifta.org" + doc.metadata.get('source', 'No source available')
        for doc in retrieved_docs
    )
    return {
        "context": context,
        "sources": sources
    }

# Answer node: uses the context and question to generate an answer.
def model_node(state: State) -> Dict[str, Any]:
    # Choose model and prompt based on language
    model_name = JAIS_MODEL if state.language == 'ar' else QWEN_MODEL
    prompt_template = PromptTemplate(
        template=JAIS_PROMPT if state.language == 'ar' else QWEN_PROMPT,
        input_variables=["context", "question", "history"]
    )
    
    prompt = prompt_template.format(
        context=state.context,
        question=state.question,
        history=state.history
    )
    
    # Initialize LLM
    llm = ChatOllama(model=model_name, temperature=TEMPERATURE, base_url=OLLAMA_BASE_URL)
    response = llm.invoke(prompt)
    
    # Add to conversation memory
    memory.add_message("user", state.question)
    memory.add_message("assistant", response.content)
    
    return {
        "response": response,
        "messages": [response]
    }

def route_model_output(state: State) -> Literal["__end__", "tools"]:
    last_message = state.messages[-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"

def ask_bot(question: str):
    # Detect language of the question
    language = detect_language(question)
    
    # Build the state graph.
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieval", retrieval_node)
    graph_builder.add_node("answer", model_node)
    graph_builder.add_node("tools", ToolNode(TOOLS))
    graph_builder.add_edge(START, "retrieval")
    graph_builder.add_edge("retrieval", "answer")
    graph_builder.add_conditional_edges("answer", route_model_output)
    graph_builder.add_edge("tools", "answer")
    graph = graph_builder.compile()

    # Get conversation history
    history = memory.get_conversation_history()

    # Define the initial state with detected language
    initial_state = State(question=question, history=history, language=language)

    # Run the graph.
    final_state = graph.invoke(initial_state)
    result = f"\nAnswer: {final_state.get('response').content}"
    sources = final_state.get('sources', [])
    if sources:
        result += "\n\nSources:"
        for source in sources:
            result += f"\n- {source}"
    return result

# for direct testing of this file
# if __name__ == "__main__":
#     print(ask_bot("should i grow my beard?"))