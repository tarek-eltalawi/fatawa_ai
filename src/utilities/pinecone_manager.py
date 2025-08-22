from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from src.utilities.utils import preprocess_text
from langchain_pinecone import PineconeVectorStore
from typing import List, Dict, Any, Optional
from langchain_core.runnables import RunnableConfig
import asyncio
from src.utilities.config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    EMBEDDING_MODEL_EN,
    EMBEDDING_MODEL_AR,
    EMBEDDING_MODEL_KWARGS,
    EMBEDDING_DIMENSION,
    TOP_K,
    PINECONE_INDEX_NAME_EN,
    PINECONE_INDEX_NAME_AR
)

class PineconeManager:
    def __init__(
        self,
        namespace: str = "qa",
        index_name: str = PINECONE_INDEX_NAME_EN,
        embedding_model: Optional[str] = None
    ):
        # Initialize Pinecone client
        self.pc = Pinecone(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT
        )
        
        self.namespace = namespace
        self.index_name = index_name
        
        # Select appropriate embedding model based on index name
        if embedding_model is None:
            embedding_model = (
                EMBEDDING_MODEL_AR 
                if index_name == PINECONE_INDEX_NAME_AR 
                else EMBEDDING_MODEL_EN
            )
        
        # Create index if it doesn't exist
        self.ensure_index_exists()
        
        # Initialize embeddings using config values
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=EMBEDDING_MODEL_KWARGS
        )
        
        self.language = 'ar' if index_name == PINECONE_INDEX_NAME_AR else 'en'
    
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
        """Create embeddings for the given texts with preprocessing."""
        # Preprocess texts based on language
        processed_texts = [
            preprocess_text(text, self.language) 
            for text in texts
        ]
        return self.embeddings.embed_documents(processed_texts)
    
    def upsert_vectors(self, vectors: List[tuple[str, List[float], dict]]):
        """Upsert vectors to Pinecone."""
        index = self.pc.Index(self.index_name)
        index.upsert(
            vectors=vectors,
            namespace=self.namespace
        )

    def retrieve_docs(self, question: str) -> Dict[str, Any]:
        """Retrieve documents with preprocessed query."""
        pinecone_index = self.pc.Index(self.index_name)
        processed_question = preprocess_text(question, self.language)
        query_vector = self.embeddings.embed_query(processed_question)
        results = pinecone_index.query(
            vector=query_vector,
            top_k=TOP_K,
            namespace=self.namespace,
            include_metadata=True,
            include_values=False
        )
        return results.matches

    def batch_fetch_vectors(self, vector_ids: List[str]) -> Dict[str, Any]:
        """Fetch multiple vectors in a single request."""
        index = self.pc.Index(self.index_name)
        try:
            result = index.fetch(
                ids=vector_ids,
                namespace=self.namespace
            )
            return result.get('vectors', {})
        except Exception as e:
            print(f"Error batch fetching vectors: {str(e)}")
            return {}

    async def aretrieve_docs(self, question: str, config: RunnableConfig) -> List[Document]:
        """Retrieve documents with preprocessed query."""
        vstore = PineconeVectorStore.from_existing_index(
            index_name=self.index_name, embedding=self.embeddings, namespace=self.namespace
        )
        retriever = vstore.as_retriever(search_kwargs={"k": TOP_K})
        return await retriever.ainvoke(question, config)
    
    async def abatch_fetch_vectors(self, vector_ids: List[str], confing: RunnableConfig) -> Dict[str, Any]:
        """Fetch multiple vectors in a single request."""
        index = self.pc.Index(self.index_name)
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, index.fetch, vector_ids, self.namespace)
            return result.vectors
        except Exception as e:
            print(f"Error batch fetching vectors: {str(e)}")
            return {}