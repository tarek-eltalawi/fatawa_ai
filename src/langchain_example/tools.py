"""
This module provides example tools for translation functionality.
"""

from typing import Any, Callable, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from deep_translator import GoogleTranslator


def translate(
    text: str, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[dict[str, Any]]:
    """This tool translates the text to Arabic."""
    try:
        translator = GoogleTranslator(source='auto', target='ar')
        translated_text = translator.translate(text)
    except Exception as err:
        # Fallback: if translation fails, return the original text with a note.
        translated_text = f"(Translation error: {err}) {text}"
    
    return {"translated_text": translated_text}

TOOLS: List[Callable[..., Any]] = [translate]
