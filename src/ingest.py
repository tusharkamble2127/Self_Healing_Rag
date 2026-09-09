from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "self_healing_rag"

DOCUMENT_TYPES = {
    "research": "research_paper",
    "company": "company_policy",
    "college": "academic_policy",
    "technical": "technical_documentation",
}


def load_documents():
    documents = []
    pdf_files = list(RAW_DATA_DIR.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {RAW_DATA_DIR}")

    for pdf_path in pdf_files:
        domain = pdf_path.parent.name
        document_type = DOCUMENT_TYPES.get(domain, "unknown")
        print(f"Loading: {pdf_path.name}")

        pages = PyPDFLoader(str(pdf_path)).load()
        for page in pages:
            page.metadata["domain"] = domain
            page.metadata["document_type"] = document_type
            page.metadata["source"] = pdf_path.name
        documents.extend(pages)

        print(f"  Domain: {domain} | Type: {document_type} | Pages: {len(pages)}")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks


def create_vector_store(chunks):
    print("\nLoading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Creating ChromaDB vector store...")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


def main():
    print("\n========== DOCUMENT INGESTION ==========\n")
    documents = load_documents()
    print(f"\nTotal pages loaded: {len(documents)}")
    chunks = split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    create_vector_store(chunks)
    print("\n========== INGESTION COMPLETE ==========")
    print(f"Vector database: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
