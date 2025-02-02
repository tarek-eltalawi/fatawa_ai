# Islamic Fatwa Assistant

A RAG (Retrieval-Augmented Generation) system that answers questions about Islamic rulings using a curated database of fatwas.

## Features
- Question answering using LLM with context from fatwa database
- Semantic search using Pinecone vector store
- Document ingestion with metadata preservation
- Deduplication of sources
- Interactive CLI interface

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```env
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=your_environment
PINECONE_INDEX_NAME=your_index_name
OLLAMA_BASE_URL=http://localhost:11434
```

3. Run Ollama locally with the deepseek model:
```bash
ollama run deepseek-r1:7b
```

## Usage
1. First ingest documents:
```bash
python ingest.py
```

2. Then run the query interface:
```bash
python query.py
```

## Upcoming Features

##### UNDERSTAND LANGCHAIN + AGENTIC AI AND APPLY IT TO THIS PROJECT.
##### ADD MULTIPLE OTHER DOCUMENTS TO THE DATABASE AND UNDERSTAND HOW TO MAKE THE SYSTEM UNDERSTAND THE DIFFERENT DOCUMENTS.
##### THIS INCLUDES UNDERSTANDING THE DIFFERENT METADATA, PROVIDE DIFFERENT SOURCES .. ETC.

#### Tools:
- [ ] Translation tool
- [ ] Web search tool
- [ ] Image search tool
- [ ] File search tool

#### Memory:
- [ ] Conversation history
- [ ] Contextual memory
- [ ] Long term memory

### High Priority
- [ ] Web interface for easier interaction
- [ ] Support for multiple languages (Arabic, Urdu, etc.)
- [ ] Question classification to improve retrieval accuracy
- [ ] Streaming responses for faster feedback
- [ ] Source ranking by relevance score

### Medium Priority
- [ ] Support for more document formats (PDF, HTML)
- [ ] Improved context handling for longer answers
- [ ] Caching frequently asked questions
- [ ] REST API endpoint for integration
- [ ] Better error handling and recovery

### Future Enhancements
- [ ] Support for conversation history
- [ ] Automated testing suite
- [ ] Documentation improvements
- [ ] User feedback collection
- [ ] Answer confidence scoring

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT
