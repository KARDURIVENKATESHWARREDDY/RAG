"""PDF ingestion utilities for the Simple Naive RAG application.

This module performs:
1) PDF loading with PyPDFLoader
2) Text chunking with RecursiveCharacterTextSplitter
3) Embedding generation with HuggingFace model
4) Storage into persistent ChromaDB
"""

from __future__ import annotations

from typing import Iterable

try:
    from langchain_chroma import Chroma
except ImportError:  # Backward compatibility if langchain-chroma isn't installed
    from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _load_pdf_documents(pdf_paths: Iterable[str]):
    """Load all pages from given PDF files into LangChain documents."""
    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs


def _split_documents(documents):
    """Split long PDF text into small chunks for better retrieval quality."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding model used for vectorization."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def ingest_pdfs(pdf_paths: list[str]) -> int:
    """Ingest PDF files into persistent ChromaDB.

    Args:
        pdf_paths: List of local PDF file paths.

    Returns:
        Number of chunks stored in ChromaDB.
    """
    if not pdf_paths:
        raise ValueError("No PDF paths were provided for ingestion.")

    # 1) Load PDFs
    documents = _load_pdf_documents(pdf_paths)
    if not documents:
        raise ValueError("No text could be loaded from the provided PDF files.")

    # 2) Chunk text
    chunks = _split_documents(documents)
    if not chunks:
        raise ValueError("Document chunking produced no chunks.")

    # 3) Create embeddings + 4) Store in Chroma
    embeddings = _get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    return len(chunks)

