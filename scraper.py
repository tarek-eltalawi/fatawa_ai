"""
Script to scrape Q&A data from sitemap and ingest it into Pinecone.
"""

import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime
import xml.etree.ElementTree as ET
from config import PINECONE_INDEX_NAME, EMBEDDING_MODEL, EMBEDDING_MODEL_KWARGS
from pinecone_manager import PineconeManager

class QAScraperPipeline:
    def __init__(self, namespace: str = "my_documents"):
        # Initialize Pinecone manager with config values
        self.pinecone_manager = PineconeManager(
            namespace=namespace,
            index_name="fatawa-in-arabic"
        )
        
        # Scraping configuration
        self.base_url = "https://www.dar-alifta.org"
        self.sitemap_url = "https://www.dar-alifta.org/sitemap/ar/sitemap.xml"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def parse_sitemap(self) -> List[Dict[str, str]]:
        """Parse sitemap and extract only fatwa URLs with their last modified dates."""
        response = requests.get(self.sitemap_url, headers=self.headers)
        
        try:
            # Find the start of the actual XML content
            xml_start = response.text.find('<urlset')
            if xml_start == -1:
                raise ValueError("Could not find XML content in response")
            
            # Extract only the XML portion
            xml_content = response.text[xml_start:]
            
            root = ET.fromstring(xml_content)
            
            # Define the namespace for cleaner code
            ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            urls = []
            # Use the namespace more cleanly
            for url in root.findall(".//sm:url", ns):
                loc = url.find("sm:loc", ns).text
                lastmod = url.find("sm:lastmod", ns).text
                
                # Only include URLs that match the fatwa format
                if "/ar/fatawa/" in loc:
                    fatwa_id = loc.split("/fatawa/")[1].split("/")[0]
                    if fatwa_id.isdigit():
                        urls.append({
                            'url': loc,
                            'lastmod': datetime.strptime(lastmod, "%Y-%m-%d"),
                            'id': fatwa_id
                        })
                    # Check if we've reached the limit
                    if len(urls) >= 10:
                        break

            return urls
        
        except Exception as e:
            print(f"Error parsing sitemap: {e}")
            print("Response content:")
            print(response.text[:500])  # Print first 500 chars for debugging
            raise

    def get_page_content(self, url: str) -> str:
        """Fetch content from a specific URL with proper Arabic encoding."""
        response = requests.get(url, headers=self.headers)
        
        # Try to get encoding from response headers
        if response.encoding:
            response.encoding = response.encoding
        # If not specified, try utf-8 (most common for Arabic web)
        else:
            response.encoding = 'utf-8'
        
        return response.text
    
    def parse_qa_from_page(self, html_content: str, url: str, qa_id: str, lastmod: datetime) -> Dict[str, str]:
        """Extract Q&A from the page."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        try:
            # Get the question: it's in the first article > p after السؤال label
            question_section = soup.find('label', class_='Questionlbl', string=lambda text: text and 'السؤال' in text)
            if question_section:
                question_div = question_section.find_next('div')
                question = question_div.find('article').find('p').get_text(strip=True)
            
            # Get the answer: it's in article#shortquestion after الجواب label
            answer_section = soup.find('article', id='shortquestion')
            if answer_section:
                # Get all p tags and combine their text
                paragraphs = answer_section.find_all('p')
                answer_texts = []
                for p in paragraphs:
                    # Skip empty paragraphs or those containing only the "details" link
                    text = p.get_text(strip=True)
                    if text and 'التفاصيل' not in text:
                        answer_texts.append(text)
                answer = ' '.join(answer_texts)
            
            if not question or not answer:
                print(f"Warning: Missing question or answer for {url}")
                return None
            
            return {
                'id': qa_id,
                'question': question,
                'answer': answer,
                'source': url,
                'last_modified': lastmod.strftime("%Y-%m-%d"),  # Convert datetime to string
                'content': f"Question: {question}\nAnswer: {answer}"
            }
        except (AttributeError, KeyError) as e:
            print(f"Error parsing QA item from {url}: {e}")
            return None
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for the given texts using PineconeManager."""
        return self.pinecone_manager.create_embeddings(texts)
    
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
                    batch[j]['id'],
                    embedding,
                    {
                        'question': batch[j]['question'],
                        'answer': batch[j]['answer'],
                        'source': batch[j]['source'],
                        'last_modified': batch[j]['last_modified']
                    }
                ))
            
            # Upload to Pinecone using manager
            self.pinecone_manager.upsert_vectors(vectors)
    
    def scrape_and_ingest(self, start_date: datetime = None):
        """Main function to scrape pages and ingest data."""
        # Get all URLs from sitemap
        urls = self.parse_sitemap()
        
        # Filter by date if start_date is provided
        if start_date:
            urls = [url for url in urls if url['lastmod'] >= start_date]
        
        all_qa_items = []
        
        with tqdm(total=len(urls), desc="Scraping pages") as pbar:
            for url_info in urls:
                try:
                    # Get page content
                    content = self.get_page_content(url_info['url'])
                    qa_item = self.parse_qa_from_page(
                        content, 
                        url_info['url'], 
                        url_info['id'],
                        url_info['lastmod']  # Pass the lastmod date
                    )
                    
                    if qa_item:
                        all_qa_items.append(qa_item)
                        print(f"Scraped item {qa_item['id']} from {url_info['url']}")
                    
                    # Add delay to be respectful to the server
                    time.sleep(2)
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error processing URL {url_info['url']}: {e}")
                    continue
        
        # Upload all items to Pinecone
        if all_qa_items:
            print(f"Uploading {len(all_qa_items)} items to Pinecone...")
            self.upload_to_pinecone(all_qa_items)
        
        print(f"Finished scraping. Total items collected: {len(all_qa_items)}")
        return all_qa_items

if __name__ == "__main__":
    # Initialize and run the scraper
    scraper = QAScraperPipeline(namespace="qa")
    # Optional: provide start_date to only scrape items modified after this date
    # start_date = datetime(2023, 1, 1)
    scraper.scrape_and_ingest() 