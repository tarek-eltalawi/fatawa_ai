import os
from typing import List, Optional
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_DIMENSION,
)
from langchain.docstore.document import Document
from pinecone_manager import PineconeManager
import json

class DocumentIngester:
    def __init__(self, namespace: str = "default"):
        # Use composition instead of inheritance
        self.pinecone_manager = PineconeManager(namespace)
        
        # Create index if it doesn't exist
        if PINECONE_INDEX_NAME not in self.pinecone_manager.pc.list_indexes().names():
            self.pinecone_manager.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def ingest_documents(self, directory_path: str, file_pattern: str = "**/*.txt") -> None:
        """
        Ingest documents from a directory into Pinecone.
        """
        # Load documents
        loader = DirectoryLoader(
            directory_path,
            glob=file_pattern,
            loader_cls=TextLoader
        )
        raw_documents = loader.load()
        
        # Process JSON content and create documents with metadata
        processed_documents = []
        for doc in raw_documents:
            content = doc.page_content
            fatwas = json.loads(content)
            for fatwa in fatwas:
                if fatwa.get("Answer"):  # Only process entries with answers
                    processed_documents.append(
                        Document(
                            page_content=fatwa["Answer"],
                            metadata={
                                "id": fatwa.get("Id"),
                                "question": fatwa.get("Question"),
                                "source": fatwa.get("Link")
                            }
                        )
                    )
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(processed_documents)
        
        # Use pinecone_manager's vector store
        self.pinecone_manager.vector_store.add_documents(texts)
        print(f"Ingested {len(texts)} text chunks into Pinecone namespace '{self.pinecone_manager.namespace}'")

if __name__ == "__main__":
    # Example usage with namespace
    ingester = DocumentIngester(namespace="my_documents")
    ingester.ingest_documents("./documents") 