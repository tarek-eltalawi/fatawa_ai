"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from react_agent import prompts
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    # Pinecone Configuration
    pinecone_api_key: str = field(default=os.getenv("PINECONE_API_KEY"))
    pinecone_environment: str = field(default=os.getenv("PINECONE_ENVIRONMENT"))
    pinecone_index_name: str = field(default=os.getenv("PINECONE_INDEX_NAME"))

    # # Ollama Configuration
    ollama_base_url: str = field(default=os.getenv("OLLAMA_BASE_URL"))
    model_name: str = field(default=os.getenv("MODEL_NAME"))
    temperature: float = field(default=float(os.getenv("TEMPERATURE")))

    # Vector Dimensions for embeddings
    embedding_dimension: int = field(default=768)

    # # Chunk size for text splitting
    chunk_size: int = field(default=1000)
    chunk_overlap: int = field(default=200)

    # # Number of relevant documents to retrieve
    top_k: int = field(default=5)

    # # Model Configuration
    embedding_model: str = field(default="sentence-transformers/all-mpnet-base-v2")
    embedding_model_kwargs: dict = field(default_factory= {"device": "cpu"})

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        config_data = {k: v for k, v in configurable.items() if k in _fields}
        if config_data.get("embedding_model_kwargs") is None:
            config_data["embedding_model_kwargs"] = {"device": "cpu"}
        return cls(**config_data)
