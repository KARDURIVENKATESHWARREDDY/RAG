"""Naive RAG pipeline functions.

This module handles:
1) Loading environment variables (including LangSmith tracing config)
2) Building embeddings + Chroma vector store
3) Retrieving similar chunks
4) Calling Ollama LLM with retrieved context
"""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# Load .env values once when this module is imported.
load_dotenv()


# Shared app constants for beginner-friendly configuration.
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "llama3"
RETRIEVAL_TOP_K = 3


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Create HuggingFace embedding model instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def _get_vector_store() -> Chroma:
    """Create or load a persistent Chroma vector store."""
    embeddings = _get_embeddings()
    return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)


def _build_prompt(context: str, question: str) -> str:
    """Build prompt for the LLM using retrieved context and user question."""
    template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant that answers questions only from the provided context.\n"
            "If the answer is not present in the context, say: 'I could not find this in the provided documents.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    )
    return template.format(context=context, question=question)


def answer_question(question: str, model_name: str = DEFAULT_OLLAMA_MODEL) -> dict[str, Any]:
    """Run naive RAG question answering.

    Steps:
    1. Retrieve top-k similar chunks from ChromaDB.
    2. Create context string from retrieved chunks.
    3. Send prompt to local Ollama model.
    4. Return answer and retrieved chunks for optional display.
    """
    vector_store = _get_vector_store()

    # Similarity search retrieval (naive retriever with k=3).
    docs = vector_store.similarity_search(question, k=RETRIEVAL_TOP_K)

    # Build context passed to LLM.
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = _build_prompt(context=context, question=question)

    # Local inference with Ollama (no cloud model).
    llm = ChatOllama(model=model_name, temperature=0)
    llm_response = llm.invoke(final_prompt)

    # Keep returned chunks in a clean structure for Streamlit UI.
    retrieved_chunks: list[dict[str, Any]] = []
    for doc in docs:
        metadata = doc.metadata or {}
        retrieved_chunks.append(
            {
                "content": doc.page_content,
                "source": metadata.get("source", "Unknown source"),
                "page": metadata.get("page", "N/A"),
            }
        )

    return {
        "answer": llm_response.content,
        "retrieved_chunks": retrieved_chunks,
    }

