import os
from typing import Optional
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
from react_agent.configuration import Configuration
from langchain_core.runnables import RunnableConfig

class PineconeManager:
    def __init__(self, namespace: str = "default", config: Optional[RunnableConfig] = None):
        configuration = Configuration.from_runnable_config(config)
        # Initialize Pinecone client
        self.pc = Pinecone(
            api_key=configuration.pinecone_api_key,
            environment=configuration.pinecone_environment
        )
        
        self.namespace = namespace
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=configuration.embedding_model,
            model_kwargs=configuration.embedding_model_kwargs
        )
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=configuration.pinecone_index_name,
            embedding=self.embeddings,
            namespace=self.namespace
        ) 