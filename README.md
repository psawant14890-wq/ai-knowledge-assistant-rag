# AI Knowledge Assistant

RAG-based knowledge assistant built with Streamlit, OpenAI, Chroma, PDF ingestion, and website ingestion.

## Features
- Multi-PDF ingestion
- Website URL ingestion
- Chroma-backed retrieval
- Chat history-aware answers
- Knowledge base summarization
- Chat summarization
- Suggested follow-up questions
- Downloadable chat transcript and source overview

## Stack
- Python
- Streamlit
- OpenAI API
- ChromaDB
- pypdf
- requests
- BeautifulSoup

## Run
1. Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` or your environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the app with `streamlit run app.py`.
