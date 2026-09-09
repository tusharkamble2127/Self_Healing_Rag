from pathlib import Path
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from retriever import retrieve_documents
from generator import generate_answer, format_context


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class CriticResult(BaseModel):
    grounded: bool = Field(
        description=(
            "True if the answer is fully supported by the retrieved "
            "context. False if the answer contains unsupported claims "
            "or the context is insufficient."
        )
    )

    confidence: float = Field(
        description="Confidence that the grounding decision is correct, from 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )

    reason: str = Field(
        description="Brief explanation of why the answer is or is not grounded."
    )


def get_critic():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Add it to the .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=api_key,
        max_retries=1,
    )

    return llm.with_structured_output(CriticResult)


def evaluate_answer(question: str, answer: str, documents):
    critic = get_critic()

    context = format_context(documents)

    prompt = f"""
You are a strict grounding evaluator for a Retrieval-Augmented
Generation system.

Your task is to determine whether the generated answer is supported
by the retrieved context.

IMPORTANT:
- Judge ONLY using the provided context.
- Do not use your own world knowledge.
- Every factual claim in the answer must be supported by the context.
- If the answer contains a claim that is not supported, mark grounded=false.
- If the context is insufficient to answer the question and the answer
  presents a definite claim anyway, mark grounded=false.
- If multiple sources contain different facts or policies, the answer
  must clearly distinguish them. A contradiction that is incorrectly
  merged should be considered ungrounded.
- Do not reward an answer merely because it sounds plausible.

USER QUESTION:
{question}

GENERATED ANSWER:
{answer}

RETRIEVED CONTEXT:
{context}

Return the grounding evaluation.
"""

    result = critic.invoke(
        [HumanMessage(content=prompt)]
    )

    return result


def main():
    question = input("\nEnter your question: ")

    print("\nRetrieving documents...")
    documents = retrieve_documents(question, k=4)

    if not documents:
        print("\nNo relevant documents were retrieved.")
        return

    print("Generating answer...")
    answer = generate_answer(question, documents)

    print("\n========== GENERATED ANSWER ==========\n")
    print(answer)

    print("\nEvaluating answer...")
    critique = evaluate_answer(
        question,
        answer,
        documents,
    )

    print("\n========== CRITIC RESULT ==========\n")
    print(f"Grounded   : {critique.grounded}")
    print(f"Confidence : {critique.confidence:.2f}")
    print(f"Reason     : {critique.reason}")


if __name__ == "__main__":
    main()