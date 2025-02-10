from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.runnables import RunnableConfig
from config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_KWARGS,
)

class PineconeManager:
    def __init__(self, namespace: str = "default", config: Optional[RunnableConfig] = None):
        # Initialize Pinecone client
        self.pc = Pinecone(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT
        )
        
        self.namespace = namespace
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=EMBEDDING_MODEL_KWARGS
        )
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=self.embeddings,
            namespace=self.namespace
        ) 