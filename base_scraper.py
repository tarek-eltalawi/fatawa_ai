"""
Base class for Q&A scrapers with common functionality.
"""

from abc import ABC, abstractmethod
import requests
from bs4 import BeautifulSoup, Tag
import time
from typing import List, Dict, Optional
from tqdm import tqdm
from datetime import datetime
import xml.etree.ElementTree as ET
from pinecone_manager import PineconeManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import json
import os

class BaseQAScraper(ABC):
    def __init__(
        self,
        batch_size: int = 50,
        sleep_time: int = 0,
        max_retries: int = 3,
        debug: bool = False
    ):
        """
        Initialize the scraper with configuration.
        
        Args:
            namespace: Pinecone namespace
            batch_size: Number of items to process in each batch
            sleep_time: Delay between requests in seconds
            max_retries: Maximum number of retry attempts for failed requests
            debug: Enable verbose logging
        """
        self.batch_size = batch_size
        self.sleep_time = sleep_time
        self.max_retries = max_retries
        self.debug = debug
        
        # Initialize common components
        self.pinecone_manager = self._init_pinecone()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Optimized separators
        )

    @abstractmethod
    def _init_pinecone(self) -> PineconeManager:
        """Initialize Pinecone with correct index name."""
        pass

    @abstractmethod
    def _get_base_url(self) -> str:
        """Return base URL for the scraper."""
        pass

    @abstractmethod
    def _get_sitemap_url(self) -> str:
        """Return sitemap URL for the scraper."""
        pass

    def parse_qa_from_page(self, html_content: str, url: str, qa_id: str, lastmod: datetime) -> Optional[Dict[str, str]]:
        """Parse Q&A content from page with common logic."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        try:
            # Find the main content container
            content_container = self._find_content_container(soup)
            if not content_container:
                self._log(f"Warning: Could not find content container for {url}")
                return None
            
            # Get question content
            question_section = self._find_question_section(content_container)
            if not question_section:
                self._log(f"Warning: Could not find question section for {url}")
                return None
            
            # Get answer content
            answer_section = self._find_answer_section(content_container)
            if not answer_section:
                self._log(f"Warning: Could not find answer section for {url}")
                return None
            
            # Extract text from sections
            question = self._extract_text_from_section(
                question_section, 
                until_tag=self._get_end_marker_for_question()
            )
            answer = self._extract_text_from_section(
                answer_section, 
                until_tag=self._get_end_marker_for_answer()
            )
            
            # Validate content
            if not self._validate_qa_content(question, answer, url):
                return None
            
            qa_item = {
                'id': qa_id,
                'question': question,
                'answer': answer,
                'source': url,
                'last_modified': lastmod.strftime("%Y-%m-%d"),
                'content': f"{self._get_question_label()}: {question}\n{self._get_answer_label()}: {answer}"
            }
            
            return qa_item
            
        except Exception as e:
            self._log(f"Error parsing QA item from {url}: {e}")
            return None

    def _get_text_from_tag(self, tag: Tag) -> str:
        """Extract and clean text from a tag."""
        return tag.get_text(strip=True) if tag else ""

    def _has_class(self, tag: Tag, class_name: str) -> bool:
        """Check if tag has specific class."""
        return class_name in tag.get('class', [])

    def _extract_text_from_section(self, section: Tag, until_tag: Optional[Dict] = None) -> str:
        """Extract text from a section until specified tag."""
        seen_content = set()
        content_parts = []
        
        current = section.find_next()
        while current:
            # Check if we've reached the end marker
            if until_tag:
                 # Convert single dict to list for uniform handling
                markers = until_tag if isinstance(until_tag, list) else [until_tag]
                
                # Check each marker - if any matches, we stop (OR logic)
                for marker in markers:
                    matches = True
                    for attr, value in marker.items():
                        current_value = current.get(attr, '')
                        if attr == 'string':
                            current_value = current.get_text(strip=True)
                        elif attr == 'name':
                            current_value = current.name
                        if not (current_value and value in current_value):
                            matches = False
                            break
                    
                    if matches:  # If any marker matches completely, stop extraction
                        return ' '.join(content_parts)
                
            if current.name == 'p':
                text = self._get_text_from_tag(current)
                if text and text not in seen_content:
                    content_parts.append(text)
                    seen_content.add(text)
                    
            current = current.find_next()
            
        return ' '.join(content_parts)

    def _validate_qa_content(self, question: str, answer: str, url: str) -> bool:
        """Validate extracted Q&A content."""
        if not question or not answer:
            self._log(f"Warning: Missing question or answer for {url}")
            return False
        return True

    @abstractmethod
    def _find_content_container(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Find the main content container."""
        pass

    @abstractmethod
    def _find_question_section(self, content: Tag) -> Optional[Tag]:
        """Find the question section within content."""
        pass

    @abstractmethod
    def _find_answer_section(self, content: Tag) -> Optional[Tag]:
        """Find the answer section within content."""
        pass

    @abstractmethod
    def _get_end_marker_for_question(self) -> Optional[Dict[str, str]]:
        """Get the marker that indicates end of question section."""
        pass

    @abstractmethod
    def _get_end_marker_for_answer(self) -> Optional[Dict[str, str]]:
        """Get the marker that indicates end of answer section."""
        pass

    @abstractmethod
    def _get_question_label(self) -> str:
        """Get the question label in the appropriate language."""
        pass

    @abstractmethod
    def _get_answer_label(self) -> str:
        """Get the answer label in the appropriate language."""
        pass

    def _log(self, message: str) -> None:
        """Log message if debug mode is enabled."""
        if self.debug:
            print(f"[DEBUG] {message}")

    def get_page_content(self, url: str) -> str:
        """Fetch content from a specific URL with retries."""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                response.encoding = 'utf-8'
                return response.text
            except requests.RequestException as e:
                self._log(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.sleep_time)

    def parse_sitemap(self) -> List[Dict[str, str]]:
        """Parse sitemap and extract fatwa URLs with their last modified dates."""
        try:
            response = requests.get(self._get_sitemap_url(), headers=self.headers)
            response.raise_for_status()
            
            # Find the start of the actual XML content
            xml_start = response.text.find('<urlset')
            if xml_start == -1:
                raise ValueError("Could not find XML content in response")
            
            xml_content = response.text[xml_start:]
            root = ET.fromstring(xml_content)
            
            ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            urls = []
            
            for url in root.findall(".//sm:url", ns):
                try:
                    loc = url.find("sm:loc", ns).text
                    lastmod = url.find("sm:lastmod", ns).text
                    
                    if self._is_valid_fatwa_url(loc):
                        fatwa_id = self._extract_fatwa_id(loc)
                        if fatwa_id:
                            urls.append({
                                'url': loc,
                                'lastmod': datetime.fromisoformat(lastmod.replace('Z', '+00:00')),
                                'id': fatwa_id
                            })

                except (AttributeError, ValueError) as e:
                    self._log(f"Error processing URL entry: {str(e)}")
                    continue
            
            return urls
            
        except Exception as e:
            self._log(f"Error parsing sitemap: {str(e)}")
            raise

    def upload_to_pinecone(self, qa_items: List[Dict[str, str]]) -> None:
        """Upload QA items to Pinecone in batches."""
        for i in range(0, len(qa_items), self.batch_size):
            batch = qa_items[i:i + self.batch_size]
            
            try:
                all_vectors = []
                for item in batch:
                    # Split answer into chunks
                    chunks = self.text_splitter.split_text(item['content'])
                    
                    # Create embeddings for chunks
                    chunk_embeddings = self.pinecone_manager.create_embeddings(chunks)
                    
                    # Create vectors for each chunk
                    for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                        vector = (
                            f"{item['id']}-{chunk_idx}",
                            embedding,
                            {
                                'text': chunk,
                                'source': item['source'],
                                'total_chunks': len(chunks)
                            }
                        )
                        all_vectors.append(vector)
                
                # Batch upload all vectors
                self.pinecone_manager.upsert_vectors(all_vectors)
                
            except Exception as e:
                self._log(f"Error uploading batch to Pinecone: {str(e)}")
                raise

    def scrape_and_ingest(self) -> List[Dict[str, str]]:
        """Main function to scrape pages and ingest data."""
        urls = self.parse_sitemap()
        
        # Load last updated time and filter URLs
        last_updated = self._load_last_updated()
        if last_updated:
            # Convert lastmod to naive datetime for comparison
            urls = [url for url in urls if url['lastmod'] and url['lastmod'].replace(tzinfo=None) >= last_updated]
            
        if not urls:
            print("No new content to scrape.")
            self.upload_existing_items()
            return []
            
        all_qa_items = []
        
        with tqdm(total=len(urls), desc="Scraping fatwas") as pbar:
            for url_info in urls:
                try:
                    content = self.get_page_content(url_info['url'])
                    qa_item = self.parse_qa_from_page(
                        content,
                        url_info['url'],
                        url_info['id'],
                        url_info['lastmod']
                    )
                    
                    if qa_item:
                        all_qa_items.append(qa_item)
                        self._log(f"Scraped fatwa {qa_item['id']} from {url_info['url']}")
                    
                    pbar.update(1)
                    
                except Exception as e:
                    self._log(f"Error processing URL {url_info['url']}: {str(e)}")
                    continue
        
        if all_qa_items:
            # Save to JSON before uploading to Pinecone
            self._save_qa_items(all_qa_items)
            
            print(f"Uploading {len(all_qa_items)} fatwas to Pinecone...")
            self.upload_to_pinecone(all_qa_items)
        
        print(f"Finished scraping. Total fatwas collected: {len(all_qa_items)}")
        return all_qa_items

    @abstractmethod
    def _is_valid_fatwa_url(self, url: str) -> bool:
        """Check if URL is a valid fatwa URL."""
        pass

    @abstractmethod
    def _extract_fatwa_id(self, url: str) -> Optional[str]:
        """Extract fatwa ID from URL."""
        pass

    def _get_json_file_path(self) -> str:
        """Get path for language-specific JSON file."""
        os.makedirs('documents', exist_ok=True)
        return f'documents/fatwas_{self.language}.json'
    
    def _load_last_updated(self) -> Optional[datetime]:
        """Load last updated time from JSON file."""
        file_path = self._get_json_file_path()
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return datetime.fromisoformat(data.get('last_updated'))
        return None
    
    def _save_qa_items(self, qa_items: List[Dict[str, str]]) -> None:
        """Save QA items to JSON file by appending new data."""
        file_path = self._get_json_file_path()
        
        # Load existing data or create new structure
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_data['data'].extend(qa_items)
                existing_data['last_updated'] = datetime.now().isoformat()
        else:
            existing_data = {
                'last_updated': datetime.now().isoformat(),
                'data': qa_items
            }
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

    def upload_existing_items(self, limit: int = 10) -> None:
        """Upload existing items from JSON file to Pinecone."""
        file_path = self._get_json_file_path()
        if not os.path.exists(file_path):
            print(f"No existing file found at {file_path}")
            return
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            items = data.get('data', [])
            
        if items:
            print(f"Uploading {len(items)} items to Pinecone...")
            self.upload_to_pinecone(items)
            print("Upload complete.")
        else:
            print("No items found in the JSON file.")