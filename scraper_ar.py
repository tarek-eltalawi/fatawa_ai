"""Arabic Q&A scraper implementation."""

from datetime import datetime
from base_scraper import BaseQAScraper
from pinecone_manager import PineconeManager
from config import PINECONE_INDEX_NAME_AR
from typing import Dict, Optional
from bs4 import BeautifulSoup, Tag

class ArabicQAScraper(BaseQAScraper):
    language = 'ar'

    def _init_pinecone(self) -> PineconeManager:
        return PineconeManager(index_name=PINECONE_INDEX_NAME_AR)

    def _get_base_url(self) -> str:
        return "https://www.dar-alifta.org"

    def _get_sitemap_url(self) -> str:
        return "https://www.dar-alifta.org/sitemap/ar/sitemap.xml"

    def _is_valid_fatwa_url(self, url: str) -> bool:
        return "/ar/fatawa/" in url

    def _extract_fatwa_id(self, url: str) -> Optional[str]:
        try:
            fatwa_id = url.split("/fatawa/")[1].split("/")[0]
            return fatwa_id if fatwa_id.isdigit() else None
        except IndexError:
            return None

    def _find_content_container(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Find the main content container for Arabic page."""
        return soup.find('body')  # Return the body tag as the container

    def _find_question_section(self, content: Tag) -> Optional[Tag]:
        """Find the question section within Arabic content."""
        return content.find('label', class_='Questionlbl', 
                          string=lambda text: text and 'السؤال' in text)

    def _find_answer_section(self, content: Tag) -> Optional[Tag]:
        """Find the answer section within Arabic content."""
        return content.find('label', string=lambda text: text and 'الجواب' in text)

    def _get_end_marker_for_question(self) -> Optional[Dict[str, str]]:
        """Get the marker that indicates end of question section."""
        return {'name': 'label', 'string': 'الجواب'}

    def _get_end_marker_for_answer(self) -> Optional[Dict[str, str]]:
        """Get the marker that indicates end of answer section."""
        return [
            {'name': 'span', 'class': 'divMoreinfo'},
            {'name': 'div', 'class': 'col-md-12'}
        ]

    def _get_question_label(self) -> str:
        return "السؤال"

    def _get_answer_label(self) -> str:
        return "الجواب"

if __name__ == "__main__":
    scraper = ArabicQAScraper(debug=True)
    scraper.scrape_and_ingest()