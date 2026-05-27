# Simple Naive RAG Chatbot (Localhost)

A beginner-friendly **Retrieval-Augmented Generation (RAG)** project using:

- Python
- LangChain
- ChromaDB
- HuggingFace embeddings
- Ollama (local LLM)
- Streamlit UI
- LangSmith tracing

This app answers questions from uploaded PDF documents using **semantic similarity search**.

---

## 1) What is RAG?

**RAG (Retrieval-Augmented Generation)** means:

1. Retrieve relevant text from your documents.
2. Send that retrieved text as context to an LLM.
3. Generate a better grounded answer.

In this project, we use a **Naive RAG pipeline** (simple and educational):

- No agents
- No memory
- No reranking
- No hybrid retrieval
- Only similarity search

---

## 2) Project Structure

```text
project_root/
│
├── data/
│   └── sample.pdf
│
├── chroma_db/
│
├── app.py
├── rag_pipeline.py
├── ingest.py
├── requirements.txt
├── .env
└── README.md
```

> Notes:
> - `data/sample.pdf` is an example location. You can place any PDF there.
> - Uploaded PDFs through the UI are also supported.

---

## 3) How the App Works

Workflow:

1. User uploads PDF(s) in Streamlit.
2. PDFs are loaded via `PyPDFLoader`.
3. Text is split using `RecursiveCharacterTextSplitter`:
   - `chunk_size=500`
   - `chunk_overlap=50`
4. Chunks are converted to embeddings using:
   - `sentence-transformers/all-MiniLM-L6-v2`
5. Embeddings are stored in persistent ChromaDB (`./chroma_db`).
6. User asks a question.
7. Retriever performs similarity search (`k=3`).
8. Retrieved context is sent to local Ollama model (`llama3`).
9. Final answer is displayed, with optional retrieved chunks.

---

## 4) LangSmith Tracing

The app supports tracing via environment variables in `.env`:

```env
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Simple-RAG
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

If you add a valid `LANGCHAIN_API_KEY`, runs can be traced in LangSmith.

---

## 5) Installation and Run (Step-by-Step Terminal Commands)

Run these commands from the project folder.

### Step 1: Create virtual environment

```bat
python -m venv .venv
```

### Step 2: Activate virtual environment

```bat
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bat
pip install -r requirements.txt
```

### Step 4: Install Ollama (if not installed)

Download and install from: https://ollama.com/download

### Step 5: Pull the Ollama model

```bat
ollama pull llama3
```

### Step 6: Make sure Ollama is running

In most setups, Ollama starts as a local service automatically.
You can test with:

```bat
ollama run llama3
```

### Step 7: Start Streamlit app

```bat
streamlit run app.py
```

---

## 6) Example Usage

1. Open the Streamlit app in your browser.
2. Upload one or more PDF files.
3. Click **Ingest Uploaded PDFs**.
4. Ask a question like:
   - `What is the main topic of this document?`
   - `Summarize chapter 2.`
5. View answer.
6. Expand **Show retrieved chunks** to inspect source context.

---

## 7) Key Files

- `app.py` → Streamlit UI
- `ingest.py` → PDF loading, chunking, embeddings, Chroma ingestion
- `rag_pipeline.py` → Similarity retrieval + prompt + Ollama answer generation

---

## 8) Beginner Notes

- This is intentionally a **simple baseline** RAG implementation.
- Good first upgrades (later) could be:
  - Better prompt formatting
  - Source citation formatting
  - Clear/reset vector DB button

For now, this project stays strictly **Naive RAG** as requested.

# RAG 
