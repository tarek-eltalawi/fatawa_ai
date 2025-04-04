import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

PINECONE_INDEX_NAME_AR = "fatawa-in-arabic"
PINECONE_INDEX_NAME_EN = "fatawa-in-english"

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# Ollama Configuration
SERVICE_HOST = os.getenv("SERVICE_HOST", "localhost")
OLLAMA_BASE_URL = f"http://{SERVICE_HOST}:11434"
QWQ_MODEL = "qwq"

REASONER_MODEL = os.getenv("REASONER_MODEL", "google/gemma-3-27b-it:free")
TOOL_CALLING_MODEL = os.getenv("TOOL_CALLING_MODEL", "google/gemini-2.5-pro-exp-03-25:free")

LOCAL_REASONER_MODEL = os.getenv("LOCAL_REASONER_MODEL", "")
LOCAL_TOOL_CALLING_MODEL = os.getenv("LOCAL_TOOL_CALLING_MODEL", "")

TEMPERATURE = 0.1

# Vector Dimensions for embeddings
EMBEDDING_DIMENSION = 768

# Chunk size for text splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CUNKS = 20

# Number of relevant documents to retrieve
TOP_K = 5

# Model Configuration
EMBEDDING_MODEL_EN = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_AR = "akhooli/Arabic-SBERT-100K"
EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}