import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
QWEN_MODEL = "qwen2.5:7b"
JAIS_MODEL = "jwnder/jais-adaptive:7b"
# MODEL_NAME = "qwen2.5:7b"
# MODEL_NAME = "jwnder/jais-adaptive:7b"
TEMPERATURE = 0.1

# Vector Dimensions for embeddings
EMBEDDING_DIMENSION = 768

# Chunk size for text splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Number of relevant documents to retrieve
TOP_K = 5

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}