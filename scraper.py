"""
Script to scrape Q&A data and ingest it into Pinecone.
"""

import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pinecone
from urllib.parse import urljoin
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QAScraperPipeline:
    def __init__(self):
        # Initialize Pinecone
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.index = pinecone.Index(self.index_name)
        
        # Initialize the embedding model
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Scraping configuration
        self.base_url = "https://www.dar-alifta.org/ar/ViewFatwa.aspx"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def get_page_content(self, page: int) -> str:
        """Fetch content from a specific page."""
        params = {'Page': str(page)}
        response = requests.get(self.base_url, params=params, headers=self.headers)
        response.encoding = 'utf-8'  # Ensure proper encoding for Arabic text
        return response.text
    
    def parse_qa_items(self, html_content: str) -> List[Dict[str, str]]:
        """Extract Q&A items from the page."""
        soup = BeautifulSoup(html_content, 'html.parser')
        qa_items = []
        
        # Update these selectors based on the actual HTML structure
        qa_containers = soup.find_all('div', class_='fatwa-container')
        
        for container in qa_containers:
            try:
                question = container.find('div', class_='question').get_text(strip=True)
                answer = container.find('div', class_='answer').get_text(strip=True)
                source = container.find('a')['href']
                
                qa_items.append({
                    'question': question,
                    'answer': answer,
                    'source': source,
                    'content': f"Question: {question}\nAnswer: {answer}"
                })
            except (AttributeError, KeyError) as e:
                print(f"Error parsing QA item: {e}")
                continue
        
        return qa_items
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for the given texts."""
        return self.model.encode(texts).tolist()
    
    def upload_to_pinecone(self, qa_items: List[Dict[str, str]], batch_size: int = 100):
        """Upload QA items to Pinecone in batches."""
        for i in range(0, len(qa_items), batch_size):
            batch = qa_items[i:i + batch_size]
            
            # Create embeddings for the batch
            texts = [item['content'] for item in batch]
            embeddings = self.create_embeddings(texts)
            
            # Prepare vectors for upload
            vectors = []
            for j, embedding in enumerate(embeddings):
                vectors.append((
                    f"qa_{i+j}",  # ID
                    embedding,    # Vector
                    {            # Metadata
                        'question': batch[j]['question'],
                        'answer': batch[j]['answer'],
                        'source': batch[j]['source']
                    }
                ))
            
            # Upload to Pinecone
            self.index.upsert(vectors=vectors)
    
    def scrape_and_ingest(self, start_page: int = 1, end_page: int = None):
        """Main function to scrape pages and ingest data."""
        current_page = start_page
        all_qa_items = []
        
        with tqdm(desc="Scraping pages") as pbar:
            while True:
                try:
                    # Get page content
                    content = self.get_page_content(current_page)
                    qa_items = self.parse_qa_items(content)
                    
                    # Break if no items found or reached end_page
                    if not qa_items or (end_page and current_page >= end_page):
                        break
                    
                    all_qa_items.extend(qa_items)
                    print(f"Scraped {len(qa_items)} items from page {current_page}")
                    
                    # Upload batch to Pinecone
                    self.upload_to_pinecone(qa_items)
                    
                    current_page += 1
                    pbar.update(1)
                    
                    # Add delay to be respectful to the server
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Error processing page {current_page}: {e}")
                    break
        
        print(f"Finished scraping. Total items collected: {len(all_qa_items)}")
        return all_qa_items

if __name__ == "__main__":
    # Initialize and run the scraper
    scraper = QAScraperPipeline()
    scraper.scrape_and_ingest(start_page=1, end_page=10)  # Adjust page range as needed 