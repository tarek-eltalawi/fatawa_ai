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
QWEN_MODEL = "qwen2.5:3b"
JAIS_MODEL = "jwnder/jais-adaptive:7b"
DEEPSEEK_MODEL = "deepseek-r1:14b"
QWQ_MODEL = "qwq"
OPENROUTER_QWQ_MODEL = "qwen/qwq-32b:free"

REASONER_MODEL = "google/gemma-3-27b-it:free"
INTERACTIVE_MODEL = "google/gemini-2.0-pro-exp-02-05:free"

LOCAL_REASONER_MODEL = os.getenv("LOCAL_REASONER_MODEL", "")
LOCAL_INTERACTIVE_MODEL = os.getenv("LOCAL_INTERACTIVE_MODEL", "")

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