from typing import Annotated, Any, Dict, List, Literal, Sequence
from dataclasses import dataclass, field
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from config import PINECONE_INDEX_NAME_AR, PINECONE_INDEX_NAME_EN, QWEN_MODEL, OLLAMA_BASE_URL, RESPONSE_LABELS, TEMPERATURE
from pinecone_manager import PineconeManager
from tools import TOOLS, detect_language
from memory import ConversationMemory
from prompts import ARABIC_PROMPT, ENGLISH_PROMPT

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
    question = state.question
    index_name = PINECONE_INDEX_NAME_EN if state.language == 'en' else PINECONE_INDEX_NAME_AR
    pinecone_manager = PineconeManager(index_name=index_name)
    
    # Get initial matches
    retrieved_docs = pinecone_manager.retrieve_docs(question)
    relevant_docs = [doc for doc in retrieved_docs if doc.score >= 0.5]
    
    # Track processed IDs and collect chunk IDs
    processed_ids = set()
    chunk_ids_to_fetch = []
    doc_chunks_map = {}  # Map base_ids to their chunks info
    
    # First pass: collect all chunk IDs we need to fetch
    for doc in relevant_docs:
        base_id = doc.id.rsplit('-', 1)[0]
        if base_id in processed_ids:
            continue
            
        processed_ids.add(base_id)
        total_chunks = int(doc.metadata.get('total_chunks', 1))
        
        # Store the first chunk we already have
        current_chunk_idx = int(doc.id.rsplit('-', 1)[1])
        doc_chunks_map[base_id] = {
            'total': total_chunks,
            'chunks': {current_chunk_idx: doc.metadata['text']},
            'source': doc.metadata.get('source', 'No source available')  # Store source with the document
        }
        
        # Collect other chunk IDs we need
        for i in range(total_chunks):
            if i != current_chunk_idx:
                chunk_ids_to_fetch.append(f"{base_id}-{i}")
    
    # Batch fetch all needed chunks
    if chunk_ids_to_fetch:
        fetched_vectors = pinecone_manager.batch_fetch_vectors(chunk_ids_to_fetch)
        
        # Process fetched chunks
        for vector_id, vector_data in fetched_vectors.items():
            base_id, chunk_idx = vector_id.rsplit('-', 1)
            if base_id in doc_chunks_map:
                doc_chunks_map[base_id]['chunks'][int(chunk_idx)] = vector_data.metadata['text']
    
    # Assemble complete answers
    complete_answers = []
    for base_id, chunk_info in doc_chunks_map.items():
        # Sort and combine chunks
        sorted_chunks = [
            chunk_info['chunks'][i] 
            for i in range(chunk_info['total']) 
            if i in chunk_info['chunks']
        ]
        complete_answer = ' '.join(sorted_chunks)
        
        complete_answers.append({
            'text': complete_answer,
            'source': chunk_info['source']  # Use the source stored with each document
        })
    
    return {
        "context": "\n\n".join(answer['text'] for answer in complete_answers),
        "sources": set(answer['source'] for answer in complete_answers)
    }

# Answer node: uses the context and question to generate an answer.
def model_node(state: State) -> Dict[str, Any]:
    # Choose model and prompt based on language
    model_name = QWEN_MODEL
    prompt_template = PromptTemplate(
        template=ARABIC_PROMPT if state.language == 'ar' else ENGLISH_PROMPT,
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
    labels = RESPONSE_LABELS[language]
    result = f"\n{labels['answer']}: {final_state.get('response').content}"
    sources = final_state.get('sources', [])
    if sources:
        result += f"\n\n{labels['sources']}:"
        for source in sources:
            result += f"\n- {source}"
    return result

# for direct testing of this file
# if __name__ == "__main__":
    # print(ask_bot("هل يجوز قراءة القرآن بدون وضوء؟"))
    # print(ask_bot("ما حكم إخراج زكاة المال في شكل إفطارٍ للصائمين؟"))
    # print(ask_bot("can I send Christmas greetings to Christian friends?"))