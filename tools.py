"""
This module provides language detection and translation functionality.
"""

from typing import Any, Callable, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

def detect_language(text: str) -> str:
    """Detect the language of the input text."""
    try:
        lang = detect(text)
        # For Arabic variants, normalize to 'ar'
        if lang in ['ar', 'arb', 'ara']:
            return 'ar'
        return lang
    except LangDetectException:
        return 'en'  # Default to English if detection fails

def translate(
    text: str, 
    target_lang: str = 'en',
    config: Annotated[RunnableConfig, InjectedToolArg] = None
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
