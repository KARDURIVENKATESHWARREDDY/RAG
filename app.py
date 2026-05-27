"""Streamlit frontend for a beginner-friendly Simple Naive RAG app.

This app lets users:
1) Upload PDF files
2) Ingest PDFs into a local ChromaDB vector store
3) Ask questions and get answers from local Ollama LLM using retrieved context
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from ingest import ingest_pdfs
from rag_pipeline import answer_question


st.set_page_config(page_title="Simple Naive RAG", page_icon="📄", layout="wide")


def _save_uploaded_files(uploaded_files: list[st.runtime.uploaded_file_manager.UploadedFile]) -> list[str]:
    """Save uploaded Streamlit files into temporary PDF files.

    Returns:
        List of file paths for the saved temporary PDFs.
    """
    temp_paths: list[str] = []
    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded.getbuffer())
            temp_paths.append(temp_file.name)
    return temp_paths


def main() -> None:
    """Main Streamlit app function."""
    st.title("📄 Simple Naive RAG Chatbot")
    st.caption("Ask questions from your PDFs using local Ollama + ChromaDB + LangChain")

    with st.sidebar:
        st.header("About")
        st.write(
            "This beginner-friendly app uses a **Naive RAG** pipeline:\n"
            "1. Load PDF\n"
            "2. Split into chunks\n"
            "3. Create embeddings\n"
            "4. Store in ChromaDB\n"
            "5. Retrieve similar chunks\n"
            "6. Generate answer with Ollama"
        )
        st.info("Default Ollama model: `llama3`")

    st.subheader("1) Upload PDF(s)")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Ingest Uploaded PDFs", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF before ingesting.")
        else:
            try:
                pdf_paths = _save_uploaded_files(uploaded_files)
                with st.spinner("Ingesting PDFs: loading, chunking, embedding, and storing in ChromaDB..."):
                    total_chunks = ingest_pdfs(pdf_paths)
                st.success(f"Ingestion complete. Stored {total_chunks} chunks in ChromaDB.")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.divider()

    st.subheader("2) Ask a Question")
    user_question = st.text_input("Enter your question")

    if st.button("Ask"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving relevant chunks and generating answer..."):
                    result = answer_question(user_question)

                st.markdown("### Answer")
                st.write(result["answer"])

                with st.expander("Show retrieved chunks"):
                    chunks = result.get("retrieved_chunks", [])
                    if not chunks:
                        st.write("No chunks were retrieved.")
                    else:
                        for idx, chunk in enumerate(chunks, start=1):
                            source = chunk.get("source", "Unknown source")
                            page = chunk.get("page", "N/A")
                            content = chunk.get("content", "")
                            st.markdown(f"**Chunk {idx}** | Source: `{Path(source).name}` | Page: `{page}`")
                            st.write(content)
                            st.divider()

            except Exception as exc:
                st.error(
                    "Question answering failed. Make sure you ingested PDFs first and Ollama is running.\n"
                    f"Details: {exc}"
                )


if __name__ == "__main__":
    main()

