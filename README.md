# AI Knowledge Assistant (RAG Chatbot)

A Retrieval-Augmented Generation (RAG) chatbot that allows users to query PDF documents using natural language.

## Features
- Upload PDF documents
- Ask questions based on document content
- Semantic search using embeddings
- Context-aware AI responses

## Tech Stack
- Python
- LangChain
- FAISS
- Streamlit

## Architecture
User Query → Embeddings → Vector DB (FAISS) → Retriever → LLM → Response

## Challenges Faced
- Handling large PDFs
- Optimizing chunk size
- Improving retrieval accuracy

## Future Improvements
- Multi-document support
- Chat history
- Web URL ingestion
