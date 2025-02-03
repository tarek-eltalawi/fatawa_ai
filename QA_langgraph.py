from langchain_ollama import OllamaLLM
from config import MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE, TOP_K
from pinecone_manager import PineconeManager
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

# Retrieval node: adds a "context" key based on the question.
def retrieval_node(state: dict) -> dict:
    pinecone_manager = PineconeManager("my_documents")
    question = state.get("question", "")
    retriever = pinecone_manager.vector_store.as_retriever(search_kwargs={"k": TOP_K})
    retrieved_docs = retriever.invoke(question)
    # Combine the content of retrieved docs into a single string.
    context = "\n".join(doc.page_content for doc in retrieved_docs)
    sources = set(
        doc.metadata.get('source', 'No source available')
        for doc in retrieved_docs
    )
    state["context"] = context
    state["sources"] = sources
    return state

# Answer node: uses the context and question to generate an answer.
def answer_node(state: dict) -> dict:
    llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=MODEL_NAME, temperature=TEMPERATURE)
    prompt_template = PromptTemplate(
        template="Using the following context, answer the question:\nContext: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )
    question = state.get("question", "")
    context = state.get("context", "")
    prompt = prompt_template.format(context=context, question=question)
    answer = llm.invoke(prompt)
    state["answer"] = answer
    return state

def ask_bot(question: str):
    # Build the state graph.
    graph_builder = StateGraph(dict)
    graph_builder.add_node("retrieval", retrieval_node)
    graph_builder.add_node("answer", answer_node)
    graph_builder.add_edge(START, "retrieval")
    graph_builder.add_edge("retrieval", "answer")
    graph_builder.add_edge("answer", END)
    graph = graph_builder.compile()

    # Define the initial state.
    initial_state = {"question": question}

    # Run the graph.
    final_state = graph.invoke(initial_state)
    result = f"\nAnswer: {final_state.get('answer')}"
    sources = final_state.get("sources", [])
    if sources:
        result += "\n\nSources:"
        for source in sources:
            result += f"\n- {source}"
    return result