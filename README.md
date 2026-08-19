# Yu-Gi-Oh! RAG Chatbot

An AI-powered Retrieval-Augmented Generation chatbot that answers questions about Yu-Gi-Oh! cards using semantic search and a local LLM.

<p align="center">
  <img src="static/screenshot1.png" width="45%" alt="Chatbot UI" />
  <img src="static/screenshot2.png" width="45%" alt="Chatbot Interaction" />
</p>

## Tech Stack

- **LLM orchestration:** LangChain, LangGraph (ReAct agent with tool-calling)
- **LLM inference:** Ollama (local, self-hosted) running Qwen2.5-7B-Instruct
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`, local/free)
- **Vector database:** ChromaDB (persistent, on-disk)
- **Data source:** YGOPRODeck REST API (~14,500 cards)
- **Frontend:** Streamlit

## Features

- **Hybrid retrieval** — combines semantic vector search with exact metadata filtering (archetype, card type) to avoid hallucinated results on structured queries like "list all X archetype cards"
- **Grounded generation** — prompt constraints prevent the LLM from inventing card names not present in retrieved context
- **Intent classification** — routes greetings/small talk away from the retrieval pipeline, so casual chat doesn't get forced through irrelevant card context
- **Conversation memory** — follow-up questions ("how many total?", "what about that one's ATK?") resolve using cached retrieval sources and chat history instead of blind re-retrieval
- **100% free/local** — no paid APIs; embeddings, vector DB, and LLM inference all run locally

## How it Works

This project implements a complete RAG pipeline:

1. **Data Collection & Chunking**: Card texts, effects, and stats are processed and combined into semantic chunks (one card = one chunk).
2. **Embedding**: Card data is converted into high-dimensional vectors using Sentence Transformers.
3. **Vector Database**: Embeddings and card metadata are stored locally using ChromaDB.
4. **Retrieval**: Queries are embedded and compared against the database to fetch the most relevant cards via semantic search, enhanced with archetype and type metadata filtering for exact-match queries.
5. **Generation**: A local LLM takes the retrieved context and user prompt to generate accurate, context-aware answers, grounded strictly in retrieved data.

## Setup

```
pip install -r requirements.txt
```

### 1. Fetch card data
```
python obtain_cards.py
```

### 2. Build embeddings and vector DB
```
python embedding.py
```

### 3. Run the chatbot
```
streamlit run streamlit_app.py
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) with any local model pulled. The default is `qwen2.5:7b-instruct`, but you can use any Ollama model by replacing the model name in `streamlit_app.py`:

```
llm = ChatOllama(model="your-model-here")
```