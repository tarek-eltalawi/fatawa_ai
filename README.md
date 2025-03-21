# Islamic Fatwa Assistant

A RAG (Retrieval-Augmented Generation) system that answers questions about Islamic rulings using a curated database of fatwas.

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
# Pinecone Configuration
PINECONE_API_KEY = "your_api_key"
```

4. LLM setup

You can choose to run a local model or use openrouter
### Using Local Model
1. Install Ollama locally if you don't have it and start the server:

```bash
brew install ollama
ollama serve
```

2. Choose a model and run it with Ollama locally e.g for qwq model:

```bash
ollama run qwq
```

Add these to your `.env` file:
```
LOCAL_REASONER_MODEL="qwq"
LOCAL_TOOL_CALLING_MODEL="qwq"  # this model has to support tool calling otherwise it won't work
```

### Using OpenRouter

Create OpenRouter account and get your API key.
Set the `OPENROUTER_API_KEY` environment variable to your OpenRouter API key.
Make sure the local env variables are not set.

`.env` should look like this
```
OPENROUTER_API_KEY="your_api_key"
REASONER_MODEL=<your_reasoner_model>  # e.g "google/gemma-3-27b-it:free"
TOOL_CALLING_MODEL=<your_tool_calling_model>  # e.g "openai/gpt-4o-mini" this model also has to support tool calling
```

## Usage

### Running the Application

1. Install langgraph:

```bash
pip install langgraph
```
2. Run the service:
```bash
langgraph dev
```

Follow the instructions here: https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/#local-development-server

### Running the Application in Debug mode and setting breakpoints

Follow instructions here: https://langchain-ai.github.io/langgraph/how-tos/local-studio/?h=debug#optional-attach-a-debugger

## Data Collection

### Running the Scrapers
The system includes scrapers for both Arabic and English fatwas from Dar Al-Iftaa. You can run them separately:

#### Using Python Scripts
1. For English Fatwas:
```bash
python index_graph/scraper_en.py
```

2. For Arabic Fatwas:
```bash
python index_graph/scraper_ar.py
```

## Troubleshooting

### Environment Setup
- Python 3.11 or higher is required
- Make sure all environment variables in `.env` are set

## Upcoming Features

- [x] test using a powerful llm hosted on a cloud service: now using Google Gemma hosted in OpenRouter
- [x] experiment with different models and different prompt engineering
- [x] convert the graph to be a async graph so that if more nodes are added, they can run in parallel
- [ ] add more nodes as tools to retrieve extra sources if needed
- [ ] one idea is to translate questions to be able to search in the other vector database (curently arabic vector database has much more data than english)
- [ ] add apps interface for both ios and android
- [ ] make the project more readable and modular
- [ ] convert scrapping and ingesting to a langgraph graph
- [ ] add a node to revise the model's answer based on the sources, or call the model multiple times and get the best answer
- [ ] add login/sign up and user's questions and answers history
- [ ] store chat history in the database
- [ ] need to add integration testing and canary scenarios after each release

### High Priority
- [x] Web interface for easier interaction
- [x] Support for conversation history
- [x] Support for multiple languages (Arabic)
- [x] Source ranking by relevance score
- [x] Streaming responses for faster feedback

### Medium Priority
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
