"""Vector store + RAG chain.

Uses HuggingFace sentence-transformers for embeddings, in-memory Chroma for
storage, and Gemini for generation. If no API key is configured the chain
factory returns None and the caller can fall back to similarity-only search.
"""
from __future__ import annotations

import os
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.0-flash"

_RAG_SYSTEM = (
    "You answer questions about a video using ONLY the transcript context "
    "provided. If the context doesn't contain the answer, say so plainly — "
    "do not invent. Quote sparingly; prefer concise synthesis. When useful, "
    "reference passages as [1], [2], etc., matching the source order."
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _RAG_SYSTEM + "\n\nContext:\n{context}"),
    ("human", "{question}"),
])


# ---------- Embeddings cache ----------

_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _EMBEDDINGS


# ---------- Vector store ----------

def build_vectorstore(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> Chroma:
    """Split text and build an in-memory Chroma index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("No content to index — transcript was empty.")
    return Chroma.from_texts(texts=chunks, embedding=_get_embeddings())


def search(vectorstore: Chroma, query: str, k: int = 4) -> List[str]:
    """Top-k similarity hits as plain strings."""
    docs = vectorstore.similarity_search(query, k=k)
    return [d.page_content for d in docs]


# ---------- RAG chain ----------

def make_qa_chain(vectorstore: Chroma, api_key: Optional[str] = None) -> Optional[Runnable]:
    """Build a Gemini-backed RAG chain. Returns None if no API key is available."""
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=0.2,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def _format(docs) -> str:
        return "\n\n".join(
            f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1)
        )

    return (
        {"context": retriever | _format, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


# ---------- Smoke test ----------

if __name__ == "__main__":
    sample = (
        "Machine learning enables computers to learn patterns from data. "
        "Deep learning is a subset that uses neural networks with many layers. "
        "Transformers are the dominant architecture for language tasks today."
    )
    vs = build_vectorstore(sample, chunk_size=80, chunk_overlap=20)
    print("Top hits for 'What is deep learning?':")
    for hit in search(vs, "What is deep learning?"):
        print(" -", hit)
