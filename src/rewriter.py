from pathlib import Path
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class RewrittenQuery(BaseModel):
    query: str = Field(
        description=(
            "A concise, specific search query that should retrieve "
            "better evidence for answering the original question."
        )
    )

    reason: str = Field(
        description=(
            "Brief explanation of why the rewritten query should "
            "improve retrieval."
        )
    )


def get_rewriter():
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

    return llm.with_structured_output(RewrittenQuery)


def rewrite_query(question: str, critique_reason: str):
    rewriter = get_rewriter()

    prompt = f"""
You are a query-rewriting component in a self-healing
Retrieval-Augmented Generation system.

The original query did not produce a sufficiently grounded answer.

Your task is to create a better search query that can retrieve
more relevant evidence from the document collection.

Rules:
1. Preserve the original user's intent.
2. Make the query more specific and information-rich.
3. Include important entities, concepts, conditions, dates,
   policies, or terminology when useful.
4. Do not invent facts.
5. Do not answer the question yourself.
6. Return ONLY a search-oriented rewritten query and a short reason.

ORIGINAL USER QUERY:
{question}

CRITIC FEEDBACK:
{critique_reason}

Create the improved retrieval query.
"""

    result = rewriter.invoke(
        [HumanMessage(content=prompt)]
    )

    return result


if __name__ == "__main__":
    question = input("\nOriginal question: ")
    critique_reason = input("\nCritic reason: ")

    result = rewrite_query(
        question,
        critique_reason,
    )

    print("\n========== REWRITTEN QUERY ==========\n")
    print(result.query)

    print("\n========== REASON ==========\n")
    print(result.reason)