# Islamic Fatwa Assistant

A RAG (Retrieval-Augmented Generation) system that answers questions about Islamic rulings using a curated database of fatwas.

## Features
- Question answering using LLM with context from fatwa database
- Semantic search using Pinecone vector store
- Document ingestion with metadata preservation
- Modern web interface for easy interaction
- Markdown rendering for formatted answers
- Source linking to original fatwas

## Setup
1. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`:
```env
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=your_environment
PINECONE_INDEX_NAME=your_index_name
```

4. Install Ollama locally if you don't have it

```bash
Brew install ollama
```

5. Start Ollama server

```bash
ollama serve
```

6. Run Ollama locally with the qwen2.5 model:

```bash
ollama run qwen2.5:14b
```

## Data Collection

### Running the Scrapers
The system includes scrapers for both Arabic and English fatwas from Dar Al-Iftaa. You can run them separately:

#### Using Python Scripts
1. For English Fatwas:
```bash
python scraper_en.py
```

2. For Arabic Fatwas:
```bash
python scraper_ar.py
```

#### Using Python Interactive Shell
1. For English Fatwas:
```python
from scraper_en import EnglishQAScraper

# Initialize with custom parameters (optional)
scraper = EnglishQAScraper(
    batch_size=50,     # Number of items per batch
    sleep_time=1,      # Delay between requests in seconds
    max_retries=3,     # Maximum retry attempts for failed requests
    debug=True         # Enable detailed logging
)

# Start scraping and ingestion
scraper.scrape_and_ingest()
```

2. For Arabic Fatwas:
```python
from scraper_ar import ArabicQAScraper

scraper = ArabicQAScraper(
    batch_size=50,
    sleep_time=1,
    max_retries=3,
    debug=True
)

scraper.scrape_and_ingest()
```

### Scraping Parameters
- `batch_size`: Number of items to process in each batch (default: 50)
- `sleep_time`: Delay between requests to avoid rate limiting (default: 1 second)
- `max_retries`: Number of retry attempts for failed requests (default: 3)
- `debug`: Enable verbose logging for troubleshooting (default: False)

### Data Storage
- Scraped fatwas are stored in Pinecone vector database
- English fatwas go to index: `fatawa-in-english`
- Arabic fatwas go to index: `fatawa-in-arabic`
- Each fatwa is split into chunks for efficient retrieval
- Metadata includes source URL and other relevant information

## Usage

### Running the Application
You can use either the CLI or web interface:

#### Web Interface (Recommended)
1. Start the Flask server:
```bash
flask run
```
2. Open your browser and navigate to: http://127.0.0.1:5000
3. Start asking questions in the web interface

#### CLI Interface
Run the query interface in terminal:
```bash
python query.py
```

## Troubleshooting

### Common Issues
1. **Import errors**: Make sure you have activated the virtual environment and installed all dependencies
2. **Flask not found**: Verify Flask is installed with `pip install flask flask-cors`
3. **Ollama connection error**: Ensure Ollama is running with the qwen2.5 model
4. **Pinecone errors**: Check your environment variables are set correctly

### Environment Setup
- Python 3.11 or higher is required
- Make sure all environment variables in `.env` are set
- The web interface requires a modern browser with JavaScript enabled

## Upcoming Features

- [x] get top 10 for example from vector database, this returns the top cunks, then construct the answer and choose top 3 answers as sources
- [x] handle arabic format in the answers and make it more presentable (written correctly from right to left)
- [x] support franco arabic
- [ ] we can also use top 3 for building the context but use the top 5 for sources links (if it makes sense by the llm)
- [ ] test using a powerful llm hosted on a cloud service
- [ ] experiment with different models and different prompt engineering, QWEN is doing good but not that good specially for arabic
- [ ] convert the graph to be a async graph so that if more nodes are added, they can run in parallel
- [ ] add more nodes as tools to retrieve extra sources if needed
- [ ] one idea is to translate questions to be able to search in the other vector database (curently arabic vector database has much more data than english)
- [ ] add apps interface for both ios and android
- [ ] separate scraping from ingesting logic
- [ ] make the project more readable and modular
- [ ] use pydantic AI with langgraph
- [ ] convert scrapping and ingesting to be a langgraph graph

#### Tools:
- [x] Translation tool
- [ ] Resources Fetching tool
- [ ] Image search tool
- [ ] File search tool

### High Priority
- [x] Web interface for easier interaction
- [x] Support for conversation history
- [x] Support for multiple languages (Arabic)
- [x] Source ranking by relevance score
- [ ] Question classification to improve retrieval accuracy
- [ ] Streaming responses for faster feedback

### Medium Priority
- [ ] Support for more document formats (PDF, HTML)
- [ ] Improved context handling for longer answers
- [ ] Caching frequently asked questions
- [ ] REST API endpoint for integration
- [ ] Better error handling and recovery

### Future Enhancements
- [ ] Add more documents to the database (e.g: Hadith, History, etc.)
- [ ] Automated testing suite
- [ ] Documentation improvements
- [ ] User feedback collection
- [ ] Answer confidence scoring
- [ ] Fine tune the model

Final vision of a workflow:
- Agents has a Dar Al Iftaa node which is the primary node. It has a tool for translation.
- Agent has a sources node with the following tools (maybe more):
    - Fiqh
    - Hadith
    - History
- User asks a question
- Agent uses Dar Al Iftaa QA node to answer the question
- Agent would offer to expand the answer with more sources and depending on the nature of the question, it would offer to use the sources node and know which tools to use.
- After executing all required tools, agent would return the final answer as a combination of all its observations.
- Agent translates the answer to the user's language
- If Dar Al Iftaa QA tool doesn't have the answer, agent would indicate that it doesn't know the answer and offer to search other tools and provide context instead of an answer. It would also create a question on behalf of the user and post it to Dar Al Iftaa QA tool. The question link would be included in the answer (e.g: I don't know the answer but I created a question on Dar Al Iftaa QA tool and posted it here: https://daraliftta.com/question/1234567890)
