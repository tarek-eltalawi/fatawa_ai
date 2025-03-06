from typing import Dict, Any, List, Set
from src.retrieval_graph.config import MAX_CUNKS, PINECONE_INDEX_NAME_AR, PINECONE_INDEX_NAME_EN
from src.utilities.pinecone_manager import PineconeManager
from langchain_core.runnables import RunnableConfig

def retrieve_documents(question: str, language: str = 'en') -> Dict[str, Any]:
    """
    Retrieve relevant documents from Pinecone based on a question.
    
    Args:
        question: The user's question
        language: The language code ('en' or 'ar')
        
    Returns:
        Dict containing raw_answers, context, and sources
    """
    index_name = PINECONE_INDEX_NAME_AR if language == 'ar' else PINECONE_INDEX_NAME_EN
    pinecone_manager = PineconeManager(index_name=index_name)
    
    # Get initial matches - now getting top 10 chunks
    retrieved_chunks = pinecone_manager.retrieve_docs(question)
    
    # Track processed IDs and collect chunk IDs
    chunk_ids_to_fetch, doc_chunks_map = collect_chunk_ids_to_fetch(retrieved_chunks)
    
    # Batch fetch all needed chunks
    if chunk_ids_to_fetch:
        fetched_vectors = pinecone_manager.batch_fetch_vectors(chunk_ids_to_fetch)
        
        # Process fetched chunks
        for vector_id, vector_data in fetched_vectors.items():
            doc_id, chunk_idx = vector_id.rsplit('-', 1)
            if doc_id in doc_chunks_map:
                doc_chunks_map[doc_id]['chunks'][int(chunk_idx)] = vector_data.metadata['text']
    
    complete_answers = construct_complete_answers(doc_chunks_map)
    
    return {
        "context": "\n\n".join(answer['text'] for answer in complete_answers),
        "sources": list(dict.fromkeys(answer['source'] for answer in complete_answers))
    }

async def aretrieve_documents(question: str, config: RunnableConfig, language: str) -> Dict[str, Any]:
    """
    Retrieve relevant documents from Pinecone based on a question.

    Args:
        question: The user's question
        language: The language code ('en' or 'ar')

    Returns:
        Dict containing raw_answers, context, and sources
    """
    index_name = PINECONE_INDEX_NAME_AR if language == 'ar' else PINECONE_INDEX_NAME_EN
    pinecone_manager = PineconeManager(index_name=index_name)
    retrieved_chunks = await pinecone_manager.aretrieve_docs(question, config)
    doc_ids_to_fetch = extract_doc_ids(retrieved_chunks)
    chunk_ids_to_fetch = build_chunk_ids_to_fetch(doc_ids_to_fetch)
    # Batch fetch all needed chunks
    fetched_vectors = await pinecone_manager.abatch_fetch_vectors(chunk_ids_to_fetch, config)
    return {
        "context": build_context(doc_ids_to_fetch, fetched_vectors),
        "sources": extract_sources(retrieved_chunks)
    }

def extract_doc_ids(retrieved_chunks):
    # Extracting doc_ids from the retrieved_chunks
    # below processes the async results to construct a list of document ids to fetch from pinecone
    # e.g doc_chunk_list = [{'4866': 11}}]
    # problem now is there is no way to fetch docs by ids from pinecone in async way
    doc_ids_to_fetch = []
    processed_ids: Set[str] = set()
    chunks_added = 0
    for chunk in retrieved_chunks:
        if chunks_added >= MAX_CUNKS:
            break
        doc_id = chunk.id.rsplit('-', 1)[0]
        total_chunks = int(chunk.metadata.get('total_chunks', 1))
        if doc_id not in processed_ids:
            doc_ids_to_fetch.append({doc_id: total_chunks})
            chunks_added += total_chunks
        processed_ids.add(doc_id)
    return doc_ids_to_fetch

def build_chunk_ids_to_fetch(doc_ids_to_fetch):
    # given list of doc_ids and chunks number: [{'4866': 11}}], we build a list of chunk ids to fetch from pinecone
    # e.g: ["4866-0", "4866-1", "4866-2", "4866-3", "4866-4", ..]
    chunk_ids_to_fetch: List[str] = []
    for doc_info in doc_ids_to_fetch:
        for doc_id, total_chunks in doc_info.items():
            for i in range(total_chunks):
                chunk_ids_to_fetch.append(f"{doc_id}-{i}")
    return chunk_ids_to_fetch

def collect_chunk_ids_to_fetch(retrieved_chunks):
    processed_ids: Set[str] = set()
    chunk_ids_to_fetch: List[str] = []
    doc_chunks_map: Dict[str, Dict] = {}  # Map base_ids to their chunks info
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
    return chunk_ids_to_fetch, doc_chunks_map

def construct_complete_answers(doc_chunks_map):
    complete_answers = []
    for doc_info in doc_chunks_map.items():
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
    return complete_answers

def build_context(doc_ids_to_fetch, fetched_vectors):
    complete_answers = []
    for doc_info in doc_ids_to_fetch:
        for doc_id, total_chunks in doc_info.items():
            # Initialize variables to store document information
            answer_text = ""
            source = "No source available"
            
            # Collect all chunks for this document
            chunks_text = []
            for i in range(total_chunks):
                chunk_id = f"{doc_id}-{i}"
                if chunk_id in fetched_vectors:
                    chunks_text.append(fetched_vectors[chunk_id].metadata['text'])
                    # Get source from the first chunk
                    if i == 0:
                        source = fetched_vectors[chunk_id].metadata.get('source', source)
            
            # Combine all chunks into a complete answer
            if chunks_text:
                answer_text = ' '.join(chunks_text)
                
                # Post-process to remove question part
                if "answer:" in answer_text.lower():
                    answer_text = answer_text.lower().split("answer:", 1)[1].strip()
                elif "الجواب: " in answer_text:
                    answer_text = answer_text.split("الجواب: ", 1)[1].strip()
                
                # Fix spaces
                answer_text = answer_text.replace("\xa0", " ")
                
                # Add to complete answers
                complete_answers.append({
                    'text': answer_text,
                    'source': source,
                })

    return "\n\n".join(answer['text'] for answer in complete_answers)

def extract_sources(retrieved_chunks):
    sources = []
    for chunk in retrieved_chunks:
        source = chunk.metadata.get('source', 'No source available')
        if source not in sources:
            sources.append(source)
    return sources