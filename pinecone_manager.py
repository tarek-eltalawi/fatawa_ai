from typing import Optional, List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.runnables import RunnableConfig
from config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_KWARGS,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSION
)

class PineconeManager:
    def __init__(
        self,
        namespace: str = "my_documents",
        index_name: str = PINECONE_INDEX_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        embedding_kwargs: Dict[str, Any] = EMBEDDING_MODEL_KWARGS
    ):
        # Initialize Pinecone client
        self.pc = Pinecone(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT
        )
        
        self.namespace = namespace
        self.index_name = index_name
        
        # Create index if it doesn't exist
        self.ensure_index_exists()
        
        # Initialize embeddings using config values
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=embedding_kwargs
        )
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=self.namespace
        )
    
    def ensure_index_exists(self):
        """Create index if it doesn't exist."""
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Index '{self.index_name}' created successfully")
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for the given texts."""
        return self.embeddings.embed_documents(texts)
    
    def upsert_vectors(self, vectors: List[tuple[str, List[float], dict]]):
        """Upsert vectors to Pinecone."""
        index = self.pc.Index(self.index_name)
        index.upsert(vectors=vectors) 