from typing import Annotated, Any, Dict, List, Literal, Sequence
from dataclasses import dataclass, field
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from config import PINECONE_INDEX_NAME_AR, PINECONE_INDEX_NAME_EN, QWEN_MODEL, OLLAMA_BASE_URL, TEMPERATURE
from pinecone_manager import PineconeManager
from utils import format_sources, process_context
from tools import TOOLS
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
    provider: str = field(default_factory=str)
    raw_answers: List[Dict[str, Any]] = field(default_factory=list)

def retrieval_node(state: State) -> Dict[str, Any]:
    question = state.question
    index_name = PINECONE_INDEX_NAME_EN if state.language == 'en' else PINECONE_INDEX_NAME_AR
    pinecone_manager = PineconeManager(index_name=index_name)
    
    # Get initial matches - now getting top 10 chunks
    retrieved_chunks = pinecone_manager.retrieve_docs(question)
    
    # Track processed IDs and collect chunk IDs
    processed_ids = set()
    chunk_ids_to_fetch = []
    doc_chunks_map = {}  # Map base_ids to their chunks info
    
    # First pass: collect all chunk IDs and initialize score tracking
    for chunk in retrieved_chunks:
        doc_id = chunk.id.rsplit('-', 1)[0]
        if doc_id not in processed_ids:
            processed_ids.add(doc_id)
            
            # Store the first chunk we already have
            current_chunk_idx = int(chunk.id.rsplit('-', 1)[1])
            doc_chunks_map[doc_id] = {
                'total': int(chunk.metadata.get('total_chunks', 1)),
                'chunks': {current_chunk_idx: chunk.metadata['text']},
                'source': chunk.metadata.get('source', 'No source available'),
                'score': chunk.score  # Store score directly in the document info
            }

            # Collect other chunk IDs we need
            for i in range(doc_chunks_map[doc_id]['total']):
                if i != current_chunk_idx:
                    chunk_ids_to_fetch.append(f"{doc_id}-{i}")
        else:
            # Update score if current chunk has higher score
            doc_chunks_map[doc_id]['score'] = max(doc_chunks_map[doc_id]['score'], chunk.score)
    
    # Batch fetch all needed chunks
    if chunk_ids_to_fetch:
        fetched_vectors = pinecone_manager.batch_fetch_vectors(chunk_ids_to_fetch)
        
        # Process fetched chunks
        for vector_id, vector_data in fetched_vectors.items():
            doc_id, chunk_idx = vector_id.rsplit('-', 1)
            if doc_id in doc_chunks_map:
                doc_chunks_map[doc_id]['chunks'][int(chunk_idx)] = vector_data.metadata['text']
    
    complete_answers = []
    for doc_id, doc_info in doc_chunks_map.items():
        # Sort and combine chunks
        sorted_chunks = [
            doc_info['chunks'][i]
            for i in range(doc_info['total'])
            if i in doc_info['chunks']
        ]
        complete_answer = ' '.join(sorted_chunks)

        # remove questions from the context supplied to llm
        if "answer:" in complete_answer.lower():
            complete_answer = complete_answer.lower().split("answer:", 1)[1].strip()
        elif "الجواب: " in complete_answer:
            complete_answer = complete_answer.split("الجواب: ", 1)[1].strip()
        
        # Fix spaces
        complete_answer = complete_answer.replace("\xa0", " ")
        
        complete_answers.append({
            'text': complete_answer,
            'source': doc_info['source'],
            'score': doc_info['score']
        })
    
    # Sort by score
    complete_answers.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        "raw_answers": complete_answers,
        "sources": list(dict.fromkeys(answer['source'] for answer in complete_answers))
    }

def postprocess_node(state: State) -> Dict[str, Any]:
    """Process and optimize the context before sending to the model."""
    # Process the raw answers to get optimized context
    processed_answers = process_context(state.raw_answers)
    
    # Create the final context
    return {
        "context": "\n\n".join(answer['text'] for answer in processed_answers)
    }

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

def ask_bot(question: str, language: str = 'en', provider: str = ''):
    # Build the state graph.
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("retrieval", retrieval_node)
    graph_builder.add_node("context_processor", postprocess_node)
    graph_builder.add_node("answer", model_node)
    graph_builder.add_node("tools", ToolNode(TOOLS))
    
    # Add edges
    graph_builder.add_edge(START, "retrieval")
    graph_builder.add_edge("retrieval", "context_processor")
    graph_builder.add_edge("context_processor", "answer")
    graph_builder.add_conditional_edges("answer", route_model_output)
    graph_builder.add_edge("tools", "answer")
    
    graph = graph_builder.compile()

    # Get conversation history
    history = memory.get_conversation_history()

    # Define the initial state with detected language
    initial_state = State(question=question, history=history, language=language, provider=provider)

    # Run the graph.
    final_state = graph.invoke(initial_state)
    answer = final_state.get('response').content
    sources = final_state.get('sources', [])
    
    # Return structured response
    return {
        "answer": answer,
        "sources": format_sources(sources, language == 'ar'),
        "language": language
    }

# for direct testing of this file
if __name__ == "__main__":
    # result = ask_bot("ما حكم إخراج زكاة المال في شكل إفطارٍ للصائمين؟")
    # result = ask_bot("هل يجوز قراءة القرآن بدون وضوء؟")
    # result = ask_bot("ممكن أسمع موسيقى ؟", "ar")
    # result = ask_bot("can I greet Christians?")
    result = ask_bot("can I listen to music?")
    answer = result["answer"]
    sources = result["sources"]
    print(answer)
    print(sources)
    # print(ask_bot("هل يجوز قراءة القرآن بدون وضوء؟"))
    # print(ask_bot("ما حكم إخراج زكاة المال في شكل إفطارٍ للصائمين؟"))
    # print(ask_bot("can I send Christmas greetings to Christian friends?"))
    # print(ask_bot("can I listen to music?"))