from pathlib import Path

from retriever import retrieve_documents
from generator import generate_answer
from critic import evaluate_answer
from rewriter import rewrite_query


MAX_RETRIES = 2


def run_self_healing_rag(question: str):

    current_query = question
    retry_count = 0

    while retry_count <= MAX_RETRIES:

        print(
            f"\n========== ATTEMPT {retry_count + 1} =========="
        )

        print(f"\nQuery: {current_query}")

        # 1. Retrieve
        documents = retrieve_documents(
            current_query,
            k=4,
        )

        if not documents:
            print("\nNo documents retrieved.")

            return {
                "answer": (
                    "I don't have enough information in the "
                    "available documents to answer this question."
                ),
                "retry_count": retry_count,
                "grounded": False,
            }

        # 2. Generate
        print("\nGenerating answer...")

        answer = generate_answer(
            question,
            documents,
        )

        print("\n========== ANSWER ==========\n")
        print(answer)

        # 3. Critic
        print("\nEvaluating grounding...")

        critique = evaluate_answer(
            question,
            answer,
            documents,
        )

        print("\n========== CRITIC ==========\n")
        print(f"Grounded   : {critique.grounded}")
        print(f"Confidence : {critique.confidence:.2f}")
        print(f"Reason     : {critique.reason}")

        # 4. Success
        if critique.grounded:

            return {
                "answer": answer,
                "documents": documents,
                "retry_count": retry_count,
                "grounded": True,
                "critique": critique,
            }

        # 5. Retry limit
        if retry_count >= MAX_RETRIES:

            return {
                "answer": (
                    "I don't have enough information in the "
                    "available documents to provide a reliable answer."
                ),
                "documents": documents,
                "retry_count": retry_count,
                "grounded": False,
                "critique": critique,
            }

        # 6. Rewrite query
        print("\nRewriting query...")

        rewritten = rewrite_query(
            question,
            critique.reason,
        )

        current_query = rewritten.query

        print("\n========== REWRITTEN QUERY ==========\n")
        print(current_query)

        retry_count += 1

    return {
        "answer": (
            "I don't have enough information in the "
            "available documents to answer this question."
        ),
        "retry_count": retry_count,
        "grounded": False,
    }


def main():

    question = input("\nEnter your question: ")

    result = run_self_healing_rag(question)

    print("\n\n========================================")
    print("           FINAL RESULT")
    print("========================================\n")

    print(result["answer"])

    print(
        f"\nAttempts used: {result['retry_count'] + 1}"
    )

    print(
        f"Grounded: {result['grounded']}"
    )

    if result.get("documents"):

        print("\n========== SOURCES ==========\n")

        seen = set()

        for doc in result["documents"]:

            source = doc.metadata.get(
                "source",
                "Unknown",
            )

            page = doc.metadata.get(
                "page",
                "Unknown",
            )

            key = (source, page)

            if key not in seen:

                seen.add(key)

                print(
                    f"- {source} | Page {page}"
                )


if __name__ == "__main__":
    main()