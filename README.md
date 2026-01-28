# Advanced RAG System

This project implements a production-style Retrieval Augmented Generation (RAG) system using Python.

## Features
- PDF ingestion and chunking
- Dense retrieval (FAISS)
- Lexical retrieval (BM25)
- Hybrid search
- Grounded answers with citations
- OpenAI-powered answer generation
- FastAPI endpoint

## How to run

Install dependencies:
pip install -r requirements.txt

Build the index:
python scripts/build_index.py

Ask questions:
python scripts/ask.py

Run API:
uvicorn app.main:app --reload
