"""Utility & helper functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from configuration import Configuration

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()

def load_chat_model(config: Configuration) -> BaseChatModel:
    """Load the local chat model from a fully specified name."""
    return ChatOllama(model=config.model_name, temperature=config.temperature, base_url=config.ollama_base_url)