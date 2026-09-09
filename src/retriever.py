import re

from vector_store import get_vector_store


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "what", "when", "where", "who", "why", "how",
    "do", "does", "did", "can", "could", "would",
    "should", "for", "to", "of", "in", "on", "at",
    "and", "or", "with", "from", "by", "about",
    "this", "that", "these", "those", "be", "been",
    "has", "have", "had", "it", "its", "their",
    "they", "them", "than", "which"
}

MIN_RERANK_SCORE = 0.32

# Runtime vector store used by the Streamlit upload workflow.
# When None, the existing persistent ChromaDB is used.
_ACTIVE_VECTOR_STORE = None


def set_active_vector_store(vector_store):
    global _ACTIVE_VECTOR_STORE
    _ACTIVE_VECTOR_STORE = vector_store


def clear_active_vector_store():
    global _ACTIVE_VECTOR_STORE
    _ACTIVE_VECTOR_STORE = None


def get_active_vector_store():
    if _ACTIVE_VECTOR_STORE is not None:
        return _ACTIVE_VECTOR_STORE

    return get_vector_store()


def tokenize(text: str):
    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )

    return {
        word
        for word in words
        if len(word) > 2 and word not in STOPWORDS
    }


def lexical_overlap(query: str, document_text: str):
    query_terms = tokenize(query)
    document_terms = tokenize(document_text)

    if not query_terms:
        return 0.0

    overlap = query_terms.intersection(document_terms)

    return len(overlap) / len(query_terms)


def normalize_distance(distance: float):
    distance = max(float(distance), 0.0)
    return 1.0 / (1.0 + distance)


def rerank_documents(query: str, results, top_k: int = 4):
    ranked = []

    for doc, distance in results:
        semantic_score = normalize_distance(
            distance
        )

        overlap_score = lexical_overlap(
            query,
            doc.page_content,
        )

        combined_score = (
            0.75 * semantic_score
            + 0.25 * overlap_score
        )

        doc.metadata["retrieval_distance"] = float(
            distance
        )

        doc.metadata["semantic_score"] = round(
            semantic_score,
            4,
        )

        doc.metadata["lexical_overlap"] = round(
            overlap_score,
            4,
        )

        doc.metadata["rerank_score"] = round(
            combined_score,
            4,
        )

        ranked.append(
            (combined_score, doc)
        )

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        doc
        for _, doc in ranked[:top_k]
    ]


def retrieve_documents(
    query: str,
    k: int = 4,
    candidate_k: int = 8,
    domain: str | None = None,
):
    vector_store = get_active_vector_store()

    if domain and domain.lower() != "all":
        # UI uses title case; stored metadata uses lowercase.
        normalized_domain = domain.strip().lower()

        results = vector_store.similarity_search_with_score(
            query,
            k=candidate_k,
            filter={"domain": normalized_domain},
        )
    else:
        results = vector_store.similarity_search_with_score(
            query,
            k=candidate_k,
        )

    if not results:
        return []

    documents = rerank_documents(
        query,
        results,
        top_k=k,
    )

    if not documents:
        return []

    best_score = max(
        float(
            doc.metadata.get(
                "rerank_score",
                0.0,
            )
        )
        for doc in documents
    )

    if best_score < MIN_RERANK_SCORE:
        return []

    return documents


def main():
    question = input("\nEnter your question: ")

    results = retrieve_documents(
        question,
        k=4,
        candidate_k=8,
    )

    if not results:
        print("\nNo sufficiently relevant documents found.")
        return

    print("\n========== RERANKED DOCUMENTS ==========\n")

    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(
            f"Source : "
            f"{doc.metadata.get('source', 'Unknown')}"
        )
        print(
            f"Domain : "
            f"{doc.metadata.get('domain', 'Unknown')}"
        )
        print(
            f"Type   : "
            f"{doc.metadata.get('document_type', 'Unknown')}"
        )
        print(
            f"Page   : "
            f"{doc.metadata.get('page', 'Unknown')}"
        )
        print(
            f"Distance       : "
            f"{doc.metadata.get('retrieval_distance', 'Unknown')}"
        )
        print(
            f"Semantic Score : "
            f"{doc.metadata.get('semantic_score', 'Unknown')}"
        )
        print(
            f"Lexical Score  : "
            f"{doc.metadata.get('lexical_overlap', 'Unknown')}"
        )
        print(
            f"Rerank Score   : "
            f"{doc.metadata.get('rerank_score', 'Unknown')}"
        )

        print(
            f"\n{doc.page_content[:800]}"
        )

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
