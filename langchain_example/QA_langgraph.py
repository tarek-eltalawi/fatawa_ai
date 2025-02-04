from typing import Any, Dict, List
from dataclasses import dataclass, field
from langchain_ollama import ChatOllama
from config import MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE, TOP_K
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

from react_agent.pinecone_manager import PineconeManager
from react_agent.tools import TOOLS


@dataclass
class State:
    question: str = field(default_factory=str)
    context: str = field(default_factory=str)
    sources: List[str] = field(default_factory=list)
    response: str = field(default_factory=str)


# Retrieval node: adds a "context" key based on the question.
def retrieval_node(state: State) -> Dict[str, Any]:
    pinecone_manager = PineconeManager("my_documents")
    question = state.question
    retriever = pinecone_manager.vector_store.as_retriever(search_kwargs={"k": TOP_K})
    retrieved_docs = retriever.invoke(question)
    # Combine the content of retrieved docs into a single string.
    context = "\n".join(doc.page_content for doc in retrieved_docs)
    sources = set(
        doc.metadata.get('source', 'No source available')
        for doc in retrieved_docs
    )
    return {
        "context": context,
        "sources": sources
    }

# Answer node: uses the context and question to generate an answer.
def model_node(state: State) -> Dict[str, Any]:
    llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE, base_url=OLLAMA_BASE_URL)
    llm.bind_tools(TOOLS)
    prompt_template = PromptTemplate(
        template="Using the following context, answer the question:\nContext: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )
    prompt = prompt_template.format(context=state.context, question=state.question)
    response = llm.invoke(prompt)
    return {
        "response": response
    }

def ask_bot(question: str):
    # Build the state graph.
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieval", retrieval_node)
    graph_builder.add_node("answer", model_node)
    graph_builder.add_edge(START, "retrieval")
    graph_builder.add_edge("retrieval", "answer")
    graph_builder.add_edge("answer", END)
    graph = graph_builder.compile()

    # Define the initial state.
    initial_state = State(question=question)

    # Run the graph.
    final_state = graph.invoke(initial_state)
    result = f"\nAnswer: {final_state.get('response').content}"
    sources = final_state.get('sources', [])
    if sources:
        result += "\n\nSources:"
        for source in sources:
            result += f"\n- {source}"
    return result

if __name__ == "__main__":
    print(ask_bot("should i grow my beard?"))