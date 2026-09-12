# 🏭 Warehouse Inventory Chatbot

> **An AI-powered RAG chatbot that lets warehouse managers query inventory data in plain English — no SQL, no dashboards, just ask.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?logo=chainlink)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-Qwen3--32B-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector--Store-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline on top of warehouse inventory CSV data. Users can ask natural language questions like:

- *"Which items are out of stock?"*
- *"What is the total available quantity by category?"*
- *"Show me all items from a specific supplier."*

The system retrieves the most relevant inventory records using **semantic search (FAISS)**, then generates a precise answer using **Qwen3-32B via Groq's free API** — all wrapped in a clean **Streamlit** web interface.

---

## 🎯 Key Features

- 📂 **Upload any inventory CSV** — no fixed schema required
- 🔍 **Semantic search** using HuggingFace embeddings (runs locally)
- ⚡ **Fast answers** via Groq's LPU inference (Qwen3-32B, free tier)
- 🧠 **Hybrid answering** — pandas for factual queries, LLM for open-ended ones
- 📊 **Live metrics dashboard** — total items, available stock, low/out of stock count
- 💬 **Persistent chat history** within the session
- 🔎 **Source transparency** — shows which chunks were retrieved per answer
- 🆓 **100% free to run** — no paid APIs required

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                             │
│         (File Upload · Chat Interface · Metric Cards)           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INDEXING PIPELINE                           │
│                                                                 │
│   CSV File                                                      │
│      │                                                          │
│      ▼                                                          │
│   Pandas ──► Enrich (stock_status, available qty)               │
│      │                                                          │
│      ▼                                                          │
│   LangChain Document Converter                                  │
│   (Row → human-readable sentence + category summaries)          │
│      │                                                          │
│      ▼                                                          │
│   RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)   │
│      │                                                          │
│      ▼                                                          │
│   HuggingFace Embeddings (all-MiniLM-L6-v2) ──► FAISS Index    │
│   [Runs locally — no API call]                                  │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                             │
│                                                                 │
│   User Question                                                 │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────────────────────────────────┐                  │
│   │         Hybrid Router                   │                  │
│   │                                         │                  │
│   │  Factual keywords?  ──► Pandas (instant)│                  │
│   │  (zero, low, total,                     │                  │
│   │   count, category)                      │                  │
│   │                                         │                  │
│   │  Open-ended?  ──────► RAG Pipeline      │                  │
│   └─────────────────────────────────────────┘                  │
│                               │                                 │
│                               ▼                                 │
│                    FAISS Retriever (top-k=10)                   │
│                               │                                 │
│                               ▼                                 │
│                    Prompt Template + Context                    │
│                               │                                 │
│                               ▼                                 │
│                    Groq API (Qwen3-32B)  ◄── API Call           │
│                               │                                 │
│                               ▼                                 │
│                         Answer + Sources                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 RAG Flow Diagram

```
                    ┌──────────────┐
                    │  inventory   │
                    │   .csv file  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    Pandas    │
                    │   Enrichment │◄── Adds: stock_status
                    └──────┬───────┘         available_qty
                           │
               ┌───────────▼───────────┐
               │  LangChain Documents  │
               │  (1 doc per row +     │
               │   category summaries) │
               └───────────┬───────────┘
                           │
               ┌───────────▼───────────┐
               │   Text Splitter       │
               │  chunk=500, overlap=50│
               └───────────┬───────────┘
                           │
               ┌───────────▼───────────┐
               │  HuggingFace Embed    │◄── all-MiniLM-L6-v2
               │  (LOCAL, no API)      │    runs on CPU
               └───────────┬───────────┘
                           │
               ┌───────────▼───────────┐
               │    FAISS Vector DB    │◄── in-memory, instant
               │    (persisted local)  │
               └───────────┬───────────┘
                           │
    User Query ────────────┤
                           │
               ┌───────────▼───────────┐
               │   Hybrid Router       │
               │  keyword → pandas     │
               │  open-ended → LLM     │
               └─────┬─────────┬───────┘
                     │         │
            ┌────────▼──┐  ┌───▼──────────────┐
            │  Pandas   │  │  FAISS Retrieval  │
            │  (instant)│  │   top-10 chunks   │
            └────────┬──┘  └───┬──────────────┘
                     │         │
                     │  ┌──────▼──────────────┐
                     │  │   Prompt Template   │
                     │  │  context + question │
                     │  └──────┬──────────────┘
                     │         │
                     │  ┌──────▼──────────────┐
                     │  │  Groq API           │◄── Free tier
                     │  │  Qwen3-32B          │    14,400 req/day
                     │  └──────┬──────────────┘
                     │         │
                     └────►────┘
                           │
                    ┌──────▼───────┐
                    │    Answer    │
                    │  + Sources   │
                    └──────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose | Cost |
|-----------|-----------|---------|------|
| **UI** | Streamlit | Web interface | Free |
| **Orchestration** | LangChain 0.2+ | RAG pipeline | Free |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Semantic search | Free (local) |
| **Vector Store** | FAISS | Similarity search | Free (local) |
| **LLM** | Qwen3-32B via Groq | Answer generation | Free tier |
| **Data** | Pandas | CSV enrichment | Free |
| **Text Splitting** | LangChain Text Splitters | Chunking | Free |

---

## 📁 Project Structure

```
warehouse-inventory-chatbot/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed to git)
├── .gitignore              # Excludes .env and cache files
│
└── data/
    └── inventory.csv       # Sample inventory data
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Free Groq API key → [console.groq.com](https://console.groq.com)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/warehouse-inventory-chatbot.git
cd warehouse-inventory-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Run the app
streamlit run app.py
```

### Usage
1. Open `http://localhost:8501` in your browser
2. Enter your Groq API key in the sidebar
3. Upload your inventory CSV or use the sample data
4. Start asking questions in plain English

---

## 💬 Example Queries

| Question | Answer Type |
|----------|------------|
| "Which items are out of stock?" | Pandas (instant) |
| "How many total items are available?" | Pandas (instant) |
| "What is the breakdown by category?" | Pandas (instant) |
| "Which products haven't been sold recently?" | LLM via RAG |
| "Suggest which items to reorder first" | LLM via RAG |
| "What items does supplier X provide?" | LLM via RAG |

---

## 🧠 How RAG Works Here

Traditional search matches keywords. This system uses **semantic similarity** — so asking *"items running low"* finds records even if they say *"quantity below threshold"*, because the meaning is similar even if the words differ.

1. **Indexing** — each CSV row is converted to a natural language sentence and embedded into a 384-dimensional vector
2. **Retrieval** — the user's question is embedded the same way; FAISS finds the 10 most similar chunks in milliseconds
3. **Generation** — those chunks are injected into a prompt and sent to Qwen3-32B, which synthesizes a precise answer

---

## ⚙️ Configuration

In `app.py`, you can tune:

```python
LLM_MODEL   = "qwen/qwen3-32b"          # swap to any Groq model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local embeddings
# In build_chain():
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # chunks retrieved
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

**Irfan Shaik**
AI Engineer · [LinkedIn](https://linkedin.com/in/YOUR_HANDLE) · [GitHub](https://github.com/YOUR_USERNAME)

> *Built as part of an AI engineering portfolio to demonstrate RAG pipeline implementation using LangChain, vector search, and LLM integration.*
