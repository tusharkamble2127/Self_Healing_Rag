from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Supported project domains.
DOMAIN_KEYWORDS = {
    "research": {
        "research", "paper", "abstract", "methodology", "experiment",
        "dataset", "results", "citation", "hypothesis", "literature",
        "model", "algorithm", "evaluation", "arxiv"
    },
    "company": {
        "employee", "employment", "hr", "human resources", "policy",
        "leave", "salary", "benefits", "workplace", "harassment",
        "organization", "company", "staff", "grievance", "insurance"
    },
    "college": {
        "student", "university", "college", "semester", "attendance",
        "examination", "scholarship", "course", "faculty", "academic",
        "admission", "hostel", "department", "curriculum"
    },
    "technical": {
        "api", "software", "database", "sql", "python", "programming",
        "documentation", "administrator", "configuration", "server",
        "query", "function", "syntax", "installation", "developer",
        "mysql", "linux", "framework"
    },
}


_embeddings = None


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    return _embeddings


def detect_domain(filename: str, text: str) -> str:
    """
    Lightweight local domain detection.
    Uses filename + first portion of document text.
    It does not call an external API.
    """
    combined = f"{filename} {text[:6000]}".lower()

    scores = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            if keyword in combined:
                # Exact multi-word phrases are slightly stronger.
                score += 2 if " " in keyword else 1

        scores[domain] = score

    best_domain, best_score = max(
        scores.items(),
        key=lambda item: item[1],
    )

    if best_score == 0:
        return "general"

    return best_domain


def load_uploaded_pdfs(uploaded_files: Iterable):
    """
    Convert Streamlit UploadedFile objects into LangChain Documents.

    Each page receives stable metadata:
    source, page, domain, document_type.
    """
    documents = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()

        if not file_bytes:
            continue

        temp_dir = BASE_DIR / ".runtime_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            uploaded_file.name,
        )

        temp_path = temp_dir / safe_name
        temp_path.write_bytes(file_bytes)

        try:
            loader = PyPDFLoader(str(temp_path))
            pages = loader.load()

            full_text = "\n".join(
                page.page_content
                for page in pages
            )

            domain = detect_domain(
                uploaded_file.name,
                full_text,
            )

            document_type_map = {
                "research": "research_paper",
                "company": "company_policy",
                "college": "academic_policy",
                "technical": "technical_documentation",
                "general": "general_document",
            }

            document_type = document_type_map.get(
                domain,
                "general_document",
            )

            for page_index, page in enumerate(pages):
                # PyPDF reports zero-based page numbers in metadata.
                # Store a user-friendly one-based page number.
                page.metadata["page"] = page_index + 1
                page.metadata["page_label"] = str(page_index + 1)
                page.metadata["source"] = uploaded_file.name
                page.metadata["domain"] = domain
                page.metadata["document_type"] = document_type

            documents.extend(pages)

        finally:
            try:
                temp_path.unlink()
            except OSError:
                pass

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


def build_uploaded_vector_store(documents):
    """
    Build an in-memory Chroma collection for the current Streamlit
    session. Nothing is written to the persistent project database.
    """
    if not documents:
        raise ValueError("No readable PDF pages were found.")

    chunks = split_documents(documents)

    collection_name = (
        "runtime_kb_"
        + re.sub(r"[^a-zA-Z0-9_-]", "", str(id(chunks)))
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
    )

    return vector_store, chunks


def summarize_chunks(chunks):
    summary = {}

    for chunk in chunks:
        domain = chunk.metadata.get(
            "domain",
            "general",
        )

        summary[domain] = (
            summary.get(domain, 0) + 1
        )

    return summary
