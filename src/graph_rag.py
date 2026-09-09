from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from retriever import retrieve_documents
from generator import generate_answer
from critic import evaluate_answer
from rewriter import rewrite_query


MAX_RETRIES = 2


class RAGState(TypedDict, total=False):
    question: str
    current_query: str
    documents: list
    answer: str

    grounded: bool
    confidence: float
    critique_reason: str

    retry_count: int
    final_response: str
    trace: list

    # Optional UI/domain filter.
    domain: str | None


def add_trace(state: RAGState, event: dict):
    trace = list(state.get("trace", []))
    trace.append(event)
    return trace


def retrieve_node(state: RAGState):
    query = state["current_query"]
    domain = state.get("domain")

    print("\n========== RETRIEVAL ==========")
    print(f"Query: {query}")

    if domain and domain != "All":
        print(f"Domain filter: {domain}")

    documents = retrieve_documents(
        query,
        k=4,
        candidate_k=8,
        domain=domain,
    )

    if documents:
        print(
            f"Relevant documents: {len(documents)}"
        )
    else:
        print(
            "No sufficiently relevant documents found."
        )

    scores = [
        doc.metadata.get("rerank_score")
        for doc in documents
    ]

    return {
        "documents": documents,
        "trace": add_trace(
            state,
            {
                "step": "retrieval",
                "status": (
                    "success"
                    if documents
                    else "insufficient"
                ),
                "query": query,
                "domain": domain or "All",
                "documents_retrieved": len(documents),
                "scores": scores,
            },
        ),
    }


def generate_node(state: RAGState):
    print("\n========== GENERATION ==========")

    if not state.get("documents"):
        return {
            "answer": "",
            "trace": add_trace(
                state,
                {
                    "step": "generation",
                    "status": "skipped",
                    "reason": (
                        "No sufficiently relevant documents"
                    ),
                },
            ),
        }

    answer = generate_answer(
        state["question"],
        state["documents"],
    )

    print("\nGenerated answer:\n")
    print(answer)

    return {
        "answer": answer,
        "trace": add_trace(
            state,
            {
                "step": "generation",
                "status": "success",
            },
        ),
    }


def critic_node(state: RAGState):
    print("\n========== CRITIC ==========")

    if (
        not state.get("documents")
        or not state.get("answer")
    ):
        return {
            "grounded": False,
            "confidence": 1.0,
            "critique_reason": (
                "The retriever did not find sufficiently "
                "relevant evidence for this question."
            ),
            "trace": add_trace(
                state,
                {
                    "step": "critic",
                    "status": "skipped",
                    "grounded": False,
                    "confidence": 1.0,
                },
            ),
        }

    critique = evaluate_answer(
        state["question"],
        state["answer"],
        state["documents"],
    )

    print(
        f"Grounded   : {critique.grounded}"
    )
    print(
        f"Confidence : {critique.confidence:.2f}"
    )
    print(
        f"Reason     : {critique.reason}"
    )

    return {
        "grounded": critique.grounded,
        "confidence": critique.confidence,
        "critique_reason": critique.reason,
        "trace": add_trace(
            state,
            {
                "step": "critic",
                "status": "success",
                "grounded": critique.grounded,
                "confidence": critique.confidence,
                "reason": critique.reason,
            },
        ),
    }


def critique_router(state: RAGState):
    if state.get("grounded", False):
        return "accepted"

    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "failed"

    return "rewrite"


def rewrite_node(state: RAGState):
    print("\n========== SELF-HEALING ==========")

    reason = state.get(
        "critique_reason",
        "The retrieved evidence was insufficient "
        "or poorly matched to the user's question.",
    )

    rewritten = rewrite_query(
        state["question"],
        reason,
    )

    new_query = rewritten.query
    retry_count = (
        state.get("retry_count", 0) + 1
    )

    print("\nRewritten query:")
    print(new_query)

    return {
        "current_query": new_query,
        "retry_count": retry_count,
        "grounded": False,
        "trace": add_trace(
            state,
            {
                "step": "query_rewrite",
                "status": "success",
                "new_query": new_query,
                "reason": rewritten.reason,
                "retry_count": retry_count,
            },
        ),
    }


def success_node(state: RAGState):
    print(
        "\n========== ANSWER VERIFIED =========="
    )

    return {
        "final_response": state["answer"],
        "trace": add_trace(
            state,
            {
                "step": "final",
                "status": "verified",
                "grounded": True,
            },
        ),
    }


def failure_node(state: RAGState):
    print(
        "\n========== INSUFFICIENT INFORMATION =========="
    )

    message = (
        "I don't have enough information in the "
        "available documents to provide a reliable answer."
    )

    return {
        "final_response": message,
        "trace": add_trace(
            state,
            {
                "step": "final",
                "status": "refused",
                "grounded": False,
            },
        ),
    }


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critic", critic_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("success", success_node)
    graph.add_node("failure", failure_node)

    graph.add_edge(
        START,
        "retrieve",
    )

    graph.add_edge(
        "retrieve",
        "generate",
    )

    graph.add_edge(
        "generate",
        "critic",
    )

    graph.add_conditional_edges(
        "critic",
        critique_router,
        {
            "accepted": "success",
            "rewrite": "rewrite",
            "failed": "failure",
        },
    )

    graph.add_edge(
        "rewrite",
        "retrieve",
    )

    graph.add_edge(
        "success",
        END,
    )

    graph.add_edge(
        "failure",
        END,
    )

    return graph.compile()


def run_graph(
    question: str,
    domain: str | None = None,
):
    app = build_graph()

    initial_state: RAGState = {
        "question": question,
        "current_query": question,
        "retry_count": 0,
        "trace": [],
        "domain": domain,
    }

    return app.invoke(
        initial_state
    )


def build_final_result(result: RAGState):
    sources = []
    seen = set()

    for doc in result.get(
        "documents",
        [],
    ):
        source = doc.metadata.get(
            "source",
            "Unknown",
        )
        page = doc.metadata.get(
            "page",
            "Unknown",
        )

        key = (source, page)

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "source": source,
                "page": page,
                "domain": doc.metadata.get(
                    "domain",
                    "Unknown",
                ),
                "document_type": doc.metadata.get(
                    "document_type",
                    "Unknown",
                ),
                "score": doc.metadata.get(
                    "rerank_score",
                    None,
                ),
            }
        )

    return {
        "question": result.get(
            "question",
            "",
        ),
        "answer": result.get(
            "final_response",
            "No final response generated.",
        ),
        "retry_count": result.get(
            "retry_count",
            0,
        ),
        "grounded": result.get(
            "grounded",
            False,
        ),
        "confidence": result.get(
            "confidence",
            0.0,
        ),
        "trace": result.get(
            "trace",
            [],
        ),
        "sources": sources,
        "domain": result.get(
            "domain",
            "All",
        ),
    }


def main():
    question = input(
        "\nEnter your question: "
    )

    result = run_graph(question)
    final_result = build_final_result(
        result
    )

    print(
        "\n\n========================================"
    )
    print(
        "            FINAL RESULT"
    )
    print(
        "========================================\n"
    )

    print(final_result["answer"])

    print(
        f"\nRetries: "
        f"{final_result['retry_count']}"
    )

    print(
        f"Grounded: "
        f"{final_result['grounded']}"
    )

    print(
        f"Confidence: "
        f"{final_result['confidence']:.2f}"
    )

    print(
        "\n========== SELF-HEALING TRACE ==========\n"
    )

    for event in final_result["trace"]:
        step = event.get(
            "step",
            "unknown",
        )
        status = event.get(
            "status",
            "unknown",
        )

        print(
            f"- {step}: {status}"
        )


if __name__ == "__main__":
    main()
