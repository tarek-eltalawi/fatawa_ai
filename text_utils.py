from typing import Optional
from pyarabic.normalize import normalize_searchtext

def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text using PyArabic.
    
    Steps:
    1. Normalize Hamza variations
    2. Normalize Lam-Alef combinations
    3. Normalize Tah Marbota
    4. Remove punctuation marks
    5. Remove extra whitespace
    """
    normalized = text
    normalized = normalize_searchtext(normalized)
    normalized = " ".join(normalized.split())  # Remove extra whitespace
    return normalized

def preprocess_text(text: str, language: Optional[str] = None) -> str:
    """
    Preprocess text based on language.
    Currently supports Arabic normalization.
    """
    if language == 'ar':
        return normalize_arabic_text(text)
    return text