"""
This module provides tools to be used by the RAG agent.
"""

from typing import Any, Callable, List, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated
from deep_translator import GoogleTranslator

def translate(
    text: str, 
    target_lang: str = 'en',
    config: Annotated[RunnableConfig, InjectedToolArg] = {}
) -> Optional[dict[str, Any]]:
    """
    NOT USED FOR NOW
    Translates text to the specified target language.
    """
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated_text = translator.translate(text)
    except Exception as err:
        # Fallback: if translation fails, return the original text with a note.
        translated_text = f"(Translation error: {err}) {text}"
    
    return {"translated_text": translated_text}

TOOLS: List[Callable[..., Any]] = []
