from typing import Optional, List, Dict, Any
from pyarabic.normalize import normalize_searchtext
from difflib import SequenceMatcher
from langchain_core.runnables import RunnableLambda
from langdetect import detect, LangDetectException
import re

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

def sources_in_markdown(sources, is_arabic=False):
    """Format sources as a list of dictionaries with titles and URLs."""
    formatted_sources = format_sources(sources, is_arabic)
    sources_title = "Sources" if not is_arabic else "المصادر"
    sources_md = "\n".join([f"- [{source['title']}]({source['url']})" for source in formatted_sources])
    return f"\n\n ##### {sources_title}:\n{sources_md}"

def format_sources(sources, is_arabic=False):
    """Format sources as a list of dictionaries with titles and URLs."""
    formatted_sources = []
    for source in sources:
        title = extract_title_from_url(source, is_arabic)
        formatted_sources.append({
            "title": title,
            "url": source
        })
    return formatted_sources

def extract_title_from_url(url, is_arabic=False):
    """Extract a readable title from the URL."""
    # Remove protocol and domain
    path = url.split('/')[-1]
    
    # Handle Arabic URLs
    if 'ar/' in url or is_arabic:
        # Replace common Arabic URL patterns
        path = path.replace('-', ' ').replace('_', ' ').replace('%20', ' ')
        return path
    
    # Handle English URLs
    title = path.split('.')[0].replace('-', ' ').replace('_', ' ')
    return title.title()

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def is_duplicate_answer(new_answer: str, existing_answers: List[str], threshold: float = 0.7) -> bool:
    """Check if an answer is too similar to existing answers."""
    for existing in existing_answers:
        if calculate_similarity(new_answer, existing) > threshold:
            return True
    return False

def count_tokens(text: str) -> int:
    """Approximate token count based on words (rough estimation)."""
    return len(text.split())

def summarize_text(text: str, max_tokens: int) -> str:
    """
    Summarize text to fit within token limit by taking the first and last parts.
    """
    return text

def detect_language(text: str) -> str:
    """Detect the language of the input text."""
    try:
        lang = detect(text)
        # For Arabic variants, normalize to 'ar'
        if lang in ['en']:
            return 'en'
        return 'ar'
    except LangDetectException:
        return 'en'  # Default to English if detection fails

def remove_chain_of_thought(message):
    # If the message is an AIMessage, extract its content and remove <think> tags
    if message.content:
        return re.sub(r"<think>.*?</think>", "", message.content, flags=re.DOTALL).strip()
    return message

# Wrap the function as a RunnableLambda for integration in the chain
cleaner = RunnableLambda(remove_chain_of_thought)

def process_context(answers: List[Dict[str, Any]], max_tokens: int = 500) -> List[Dict[str, Any]]:
    """
    Preprocess context by limiting tokens and removing duplicates.
    """
    processed_answers = []
    current_token_count = 0
    existing_texts = []
    
    for answer in answers:
        answer_text = answer['text']
        answer_tokens = count_tokens(answer_text)
        
        # If this is the first answer and it exceeds token limit, summarize it
        if not processed_answers and answer_tokens > max_tokens:
            summarized_text = summarize_text(answer_text, max_tokens)
            answer = {**answer, 'text': summarized_text}
            processed_answers.append(answer)
            break
        
        # Skip if this would exceed token limit
        if current_token_count + answer_tokens > max_tokens:
            break
            
        # Skip if too similar to existing answers
        if is_duplicate_answer(answer_text, existing_texts):
            continue
            
        processed_answers.append(answer)
        existing_texts.append(answer_text)
        current_token_count += answer_tokens
    
    return processed_answers
