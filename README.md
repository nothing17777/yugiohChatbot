# Yu-Gi-Oh! RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions about Yu-Gi-Oh! cards using semantic search and a local LLM.

## Setup

```bash
pip install -r requirements.txt
```

### 1. Fetch card data
```bash
python obtain_cards.py
```

### 2. Build embeddings and vector DB
```bash
python embedding.py
```

### 3. Run the chatbot
```bash
streamlit run streamlit_app.py
```

## Requirements
- Python 3.10+
- [Ollama](https://ollama.com/) with `qwen2.5:7b-instruct` pulled (`ollama pull qwen2.5:7b-instruct`)
