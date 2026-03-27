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

## Demo Flow
1. Launch the app and load your API key from Streamlit secrets or environment variables.
2. Upload 1-2 PDFs and optionally add a few website URLs.
3. Click `Initialize Knowledge Base`.
4. Ask a question in chat or use one of the suggested prompts.
5. Open `View sources` under an answer to show grounding and citations.
6. Generate `Summarize Knowledge Base` and `Summarize Chat`.
7. Download the transcript or source overview from the right-side tools panel.
