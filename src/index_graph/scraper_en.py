"""English Q&A scraper implementation."""

from src.index_graph.base_scraper import BaseQAScraper
from src.utilities.pinecone_manager import PineconeManager
from retrieval_graph.config import PINECONE_INDEX_NAME_EN
from typing import Dict, Optional
from bs4 import BeautifulSoup, Tag

class EnglishQAScraper(BaseQAScraper):
    language = 'en'

    def _init_pinecone(self) -> PineconeManager:
        return PineconeManager(index_name=PINECONE_INDEX_NAME_EN)

    def _get_base_url(self) -> str:
        return "https://www.dar-alifta.org"

    def _get_sitemap_url(self) -> str:
        return "https://www.dar-alifta.org/en/sitemap.xml"

    def _is_valid_fatwa_url(self, url: str) -> bool:
        return "/en/fatwa/details/" in url

    def _extract_fatwa_id(self, url: str) -> Optional[str]:
        try:
            fatwa_id = url.split("/details/")[1].split("/")[0]
            return fatwa_id if fatwa_id.isdigit() else None
        except IndexError:
            return None

    def _find_content_container(self, soup: BeautifulSoup) -> Optional[Tag]:
        """Find the main content container for English page."""
        return soup.find('div', class_='entry-content')

    def _find_question_section(self, content: Tag) -> Optional[Tag]:
        """Find the question section within English content."""
        heading_blocks = content.find_all('div', class_='heading-block')
        for block in heading_blocks:
            h4_tag = block.find('h4')
            if h4_tag and 'Question' in h4_tag.text:
                return block
        return None

    def _find_answer_section(self, content: Tag) -> Optional[Tag]:
        """Find the answer section within English content."""
        heading_blocks = content.find_all('div', class_='heading-block')
        for block in heading_blocks:
            h4_tag = block.find('h4')
            if h4_tag and 'Answer' in h4_tag.text:
                return block
        return None

    def _get_end_marker_for_question(self) -> Optional[Dict[str, str]]:
        """Get the marker that indicates end of question section."""
        return {'name': 'h4', 'string': 'Answer'}

    def _get_end_marker_for_answer(self) -> Optional[Dict[str, str]]:
        """Get the marker that indicates end of answer section."""
        return {'name': 'div', 'class': 'clear'}

    def _get_question_label(self) -> str:
        return "Question"

    def _get_answer_label(self) -> str:
        return "Answer"

if __name__ == "__main__":
    scraper = EnglishQAScraper(debug=True)
    scraper.scrape_and_ingest()