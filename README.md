# DocuChat AI

A RAG-based document Q&A chatbot that lets you upload a PDF and ask questions about it. Built this as my flagship project while learning Python and Gen AI development.

## What it does

Upload any PDF, and the app reads through it, breaks it into chunks, converts them into embeddings, and stores them in a FAISS vector store. When you ask a question, it retrieves the most relevant chunks and passes them to Gemini to generate an answer grounded in the actual document content — not just generic LLM knowledge.

There's also a Pro tier that generates a downloadable PDF report of the conversation/analysis using ReportLab.

## Tech stack

- **Python 3.11**
- **LangChain** – for chaining the retrieval + generation pipeline
- **Google Gemini API** – for embeddings and answer generation
- **FAISS** – vector storage and similarity search
- **Streamlit** – frontend/UI
- **ReportLab** – PDF report generation for the Pro tier

## Features

- Upload a PDF and chat with it directly
- Context-aware answers based only on the uploaded document
- Handles reasonably large PDFs (tested with 60+ page documents)
- Pro tier: generates a downloadable PDF summary/report of the Q&A session
- Clean Streamlit interface, no clunky setup needed to run locally

## Running it locally

Clone the repo:
```bash
git clone https://github.com/divxmishra/docuchat-ai.git
cd docuchat-ai
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file in the root folder and add your own Gemini API key:
```
GOOGLE_API_KEY=your_api_key_here
```

Run the app:
```bash
streamlit run app.py
```

## Why I built this

I wanted to actually understand how RAG pipelines work instead of just reading about them — how documents get chunked, how embeddings work, how retrieval ties into an LLM call, and how all of that gets wrapped into something usable. This project forced me to deal with real problems: embedding model changes, deprecated LangChain chains, API quota limits, and the usual git/GitHub headaches that come with managing a project properly (including making sure API keys never end up in version control).

## Note

`.env` is intentionally excluded from this repo — you'll need your own Gemini API key to run it locally.

---

Feedback and suggestions are welcome. Still actively improving this as I learn more.
