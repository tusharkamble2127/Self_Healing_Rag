from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from retriever import retrieve_documents

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Add it to the .env file.")
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
        google_api_key=api_key,
        max_retries=1,
    )


def format_context(documents):
    context_parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        domain = doc.metadata.get("domain", "Unknown")
        document_type = doc.metadata.get("document_type", "Unknown")
        score = doc.metadata.get("retrieval_score", "Unknown")
        context_parts.append(
            f"""
--- CONTEXT {i} ---

Source: {source}
Page: {page}
Domain: {domain}
Document Type: {document_type}
Retrieval Score: {score}

{doc.page_content}
"""
        )
    return "\n".join(context_parts)


def generate_answer(question: str, documents):
    llm = get_llm()
    context = format_context(documents)
    prompt = f"""
You are a retrieval-augmented generation assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the context does not contain enough information, clearly say that there is not enough information.
4. When multiple sources contain different policies or facts, explain the differences and identify the relevant source.
5. Keep the answer clear and concise.
6. Preserve important numbers, dates, names, and conditions exactly.
7. Do not claim that something is true unless supported by the context.
8. Do not add unrelated information merely because it appears in the retrieved context.
9. Only include a policy or fact when it helps answer the user's question.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Now provide the best grounded answer.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    if isinstance(response.content, str):
        return response.content
    if isinstance(response.content, list):
        text_parts = []
        for part in response.content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    text_parts.append(text)
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts).strip()
    return str(response.content)


def main():
    question = input("\nEnter your question: ")
    documents = retrieve_documents(question, k=4)
    if not documents:
        print("\nNo relevant documents were retrieved.")
        return

    answer = generate_answer(question, documents)
    print("\n========== FINAL ANSWER ==========\n")
    print(answer)
    print("\n========== SOURCES ==========\n")

    seen = set()
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            print(f"- {source} | Page {page}")


if __name__ == "__main__":
    main()
