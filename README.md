# Islamic Fatwa Assistant

A RAG (Retrieval-Augmented Generation) system that answers questions about Islamic rulings using a curated database of fatwas.

## Features
- Question answering using LLM with context from fatwa database
- Semantic search using Pinecone vector store
- Document ingestion with metadata preservation
- Deduplication of sources
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

4. Run Ollama locally with the qwen2.5 model:
```bash
ollama run qwen2.5:7b
```

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
- Python 3.8 or higher is required
- Make sure all environment variables in `.env` are set
- The web interface requires a modern browser with JavaScript enabled

## Upcoming Features

#### Tools:
- [x] Translation tool
- [ ] Web search tool
- [ ] Image search tool
- [ ] File search tool

### High Priority
- [x] Web interface for easier interaction
- [ ] Support for conversation history
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
- [ ] Add more documents to the database (e.g: Hadith, History, etc.)
- [ ] Automated testing suite
- [ ] Documentation improvements
- [ ] User feedback collection
- [ ] Answer confidence scoring

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
