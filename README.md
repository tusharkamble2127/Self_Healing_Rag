# 🧠 Self-Healing RAG

A multi-domain Retrieval-Augmented Generation (RAG) system built with **LangChain, LangGraph, ChromaDB, Hugging Face embeddings, Gemini, and Streamlit**.

Unlike a traditional RAG pipeline that simply retrieves documents and generates an answer, this system **evaluates its own answer** and can automatically **rewrite the search query, retrieve better evidence, and retry the generation** when the answer is not sufficiently grounded.

The final application also supports **runtime PDF uploads**, so users can create a temporary knowledge base directly from the UI instead of manually placing documents into a fixed folder.

-------------------------------------------------------------------------------------------------------------------------------------------------

## 📌 Project Overview

### Problem

A conventional RAG system follows:
User Question
      ↓
Retrieve Documents
      ↓
Generate Answer
      ↓
Return Answer


This has a major weakness: if retrieval is poor, the generated answer may be incomplete, unsupported, or hallucinated.

### Proposed Solution

This project adds a **grounding critic** and a **self-healing loop**:

User Question
      ↓
Retrieve Relevant Chunks
      ↓
Generate Answer
      ↓
Critic / Grounding Check
      ↓
   Is Answer Grounded?
      ├───────────────┐
      │ YES           │ NO
      ↓              ↓
 Final Answer     Rewrite Query
                      ↓
                Re-retrieve
                      ↓
                 Generate Again
                      ↓
                    Critic
                      ↺


If repeated attempts still fail, the system does not invent an answer. It safely returns:

> "I don't have enough information in the available documents to provide a reliable answer."

-------------------------------------------------------------------------------------------------------------------------------------------------

# 🎯 Objectives

The main objectives of the project are:

1. Build a reliable Retrieval-Augmented Generation system.
2. Support multiple document domains.
3. Retrieve semantically relevant document chunks.
4. Improve retrieval using local reranking.
5. Detect unsupported or hallucinated answers.
6. Automatically reformulate failed queries.
7. Retry retrieval and generation using LangGraph.
8. Refuse to answer when sufficient evidence is unavailable.
9. Show trustworthy source information with page numbers.
10. Provide a user-friendly Streamlit interface for uploading PDFs and asking questions.

-------------------------------------------------------------------------------------------------------------------------------------------------

# 🧩 Supported Knowledge Domains

The application is designed to work with mixed-domain PDF collections.

### 🔬 Research

Examples:

- Research papers
- Scientific papers
- Experimental reports
- Technical research documents

### 🏢 Company

Examples:

- Employee handbooks
- HR policies
- Leave policies
- Workplace policies
- Benefits and insurance policies

### 🎓 College

Examples:

- Student handbooks
- Academic regulations
- Attendance policies
- Scholarship policies
- Examination rules

### 💻 Technical

Examples:

- Software documentation
- Database documentation
- Administrator guides
- API documentation
- Developer manuals

The UI can search:
All
Research
Company
College
Technical

-------------------------------------------------------------------------------------------------------------------------------------------------

# 🏗️ System Architecture

## High-Level Architecture


                         ┌───────────────────────┐
                         │      Streamlit UI     │
                         └───────────┬───────────┘
                                     │
                              Upload PDF files
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    PDF Processing     │
                         │  PyPDFLoader + Split  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    Domain Detection   │
                         │ Research / Company    │
                         │ College / Technical   │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   Embedding Model     │
                         │ all-MiniLM-L6-v2      │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │       ChromaDB        │
                         │     Vector Store      │
                         └───────────┬───────────┘
                                     │
                                  Query
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │       Retriever       │
                         │ Semantic Search       │
                         │ + Lexical Reranking   │
                         │ + Relevance Threshold │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │       Generator       │
                         │      Gemini LLM       │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │      Critic Agent      │
                         │   Grounding Check      │
                         └───────────┬───────────┘
                                     │
                            ┌────────┴────────┐
                            │                 │
                         Grounded          Rejected
                            │                 │
                            ▼                 ▼
                         Success        Query Rewriter
                                              │
                                              ▼
                                         Re-retrieve
                                              │
                                              └───────↺


-------------------------------------------------------------------------------------------------------------------------------------------------

# 🔄 LangGraph Workflow

The core agentic workflow is implemented using **LangGraph** as a stateful, cyclic graph.

START
  ↓
retrieve
  ↓
generate
  ↓
critic
  ↓
┌───────────────┬────────────────┬─────────────┐
│               │                │
accepted      rewrite          failed
│               │                │
▼               ▼                ▼
success       retrieve         failure
│               ↺                │
▼                                ▼
END                              END

## Graph State

The graph maintains information such as:

```python
{
    "question": ...,
    "current_query": ...,
    "documents": ...,
    "answer": ...,
    "grounded": ...,
    "confidence": ...,
    "critique_reason": ...,
    "retry_count": ...,
    "final_response": ...,
    "trace": ...,
    "domain": ...
}
```

The original user question remains unchanged. Only the **retrieval query** can be rewritten during self-healing.

-------------------------------------------------------------------------------------------------------------------------------------------------

# 📚 Document Processing Pipeline

When PDFs are uploaded:
PDF
 ↓
Page Extraction
 ↓
Metadata Assignment
 ↓
Domain Detection
 ↓
Chunking
 ↓
Embedding Generation
 ↓
ChromaDB


## Chunking

The project currently uses:

- `RecursiveCharacterTextSplitter`
- Chunk size: approximately 1000 characters
- Chunk overlap: approximately 150 characters

The overlap helps preserve context across chunk boundaries.

-------------------------------------------------------------------------------------------------------------------------------------------------

# 🏷️ Metadata

Each document chunk stores useful metadata such as:

```python
{
    "source": "attendance_policy.pdf",
    "page": 20,
    "domain": "college",
    "document_type": "academic_policy",
    "chunk_id": 123
}
```

This metadata is used for:

- Domain filtering
- Source display
- Page references
- Debugging
- Retrieval analysis
- Evaluation

-------------------------------------------------------------------------------------------------------------------------------------------------

# 🔎 Retrieval System

The retriever does not simply take the first four vector-search results.

The current pipeline is:

Query
 ↓
ChromaDB Candidate Search
 ↓
Semantic Distance
 ↓
Normalized Semantic Score
 +
Lexical Overlap
 ↓
Combined Rerank Score
 ↓
Relevance Threshold
 ↓
Top Relevant Chunks

-------------------------------------------------------------------------------------------------------------------------------------------------

# 🧠 Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are generated locally.

Advantages:

- No embedding API cost
- Fast enough for development
- Works well for semantic retrieval
- Keeps the vector-search layer local

---

# 🗄️ Vector Database

The vector store is:

```text
ChromaDB
```

Two modes are supported conceptually:

### Persistent Development Mode

The original ingestion pipeline can build:

```text
chroma_db/
```

from documents stored under:

```text
data/raw/
```

### Runtime Upload Mode

The final Streamlit application builds a temporary vector database from files uploaded during the current session.

This runtime knowledge base is intentionally separated from the persistent development database.

-------------------------------------------------------------------------------------------------------------------------------------------------

# 🤖 LLM Layer

Gemini is used for:

### 1. Answer Generation

The LLM receives:

```text
User Question
+
Retrieved Context
```

and must answer only from the supplied context.

### 2. Critic Agent

The critic evaluates:

```text
Question
+
Retrieved Context
+
Generated Answer
```

and returns structured information such as:

```json
{
  "grounded": true,
  "confidence": 0.95,
  "reason": "The answer is supported by the retrieved context."
}
```

### 3. Query Rewriter

If the critic rejects the answer, the query rewriter creates a more retrieval-friendly search query.

Example:

```text
Original:
What is the minimum attendance requirement?

Rewritten:
minimum attendance requirement percentage eligibility policy
write examinations
```

---

# 🩺 Self-Healing Mechanism

The self-healing mechanism works like this:

### Attempt 1

```text
Question
 ↓
Retrieve
 ↓
Generate
 ↓
Critic
```

If grounded:

```text
→ Return answer
```

If rejected:

```text
→ Rewrite query
→ Retrieve again
→ Generate again
→ Critic again
```

### Retry Limit

Current maximum retry count:

```text
MAX_RETRIES = 2
```

Therefore the system can perform up to three retrieval/generation attempts including the initial attempt.

If all attempts fail:

```text
I don't have enough information in the available documents
to provide a reliable answer.
```

---

# 🧪 Self-Healing Verification

The self-healing mechanism was explicitly tested using a controlled development test.

The first answer was deliberately made incorrect:

```text
The minimum attendance requirement is 50%...
```

The critic detected the problem:

```text
Grounded = False
```

The system then rewrote the query:

```text
minimum attendance requirement percentage eligibility policy
write examinations
```

The second attempt retrieved better evidence and produced the correct answer:

```text
80% attendance
```

The critic then accepted the result:

```text
Grounded = True
```

This proved that the cyclic recovery workflow works as intended.

The intentional test hook was removed from the production workflow afterward.

---

# 🛡️ Safe Refusal

A major design goal is to avoid hallucination.

For questions outside the knowledge base, the retriever can reject weak matches before an LLM call.

Example:

```text
Question:
What is the current temperature on Mars?
```

Result:

```text
No sufficiently relevant documents found.
```

The LLM is not given unrelated context.

This creates:

```text
Out-of-domain Question
        ↓
Weak Retrieval
        ↓
Threshold Rejection
        ↓
Safe Refusal
```

---

# 🖥️ Streamlit Application

The final interface provides:

### Knowledge Base

Users can:

- Upload multiple PDFs
- Upload mixed-domain documents
- Process the documents
- See document/chunk counts
- See detected domains
- Clear the current knowledge base

### Question Interface

Users can:

- Ask natural-language questions
- Search all domains
- Optionally select a specific domain
- View the verified answer

### Result Information

The UI displays:

```text
Answer
Retries
Grounded
Confidence
Sources
```

The internal LangGraph trace is retained in the backend for evaluation/debugging but is intentionally hidden from the main UI.

-------------------------------------------------------------------------------------------------------------------------------------------------

# ⚙️ Installation

## 1. Clone / open the project

```powershell
cd F:\self-healing-rag-Project
```

Use your own project path if it is different.

---

## 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

You should see:

```text
(.venv)
```

in the terminal prompt.

---

## 3. Install dependencies

```powershell
python -m pip install -U pip
```

Then:

```powershell
pip install -r requirements.txt
```
-------------------------------------------------------------------------------------------------------------------------------------------------

Recommended `.gitignore` entries:

```gitignore
.venv/
.env
__pycache__/
*.pyc
.pytest_cache/
chroma_db/
data/processed/
.runtime_uploads/
```

---

# ▶️ Running the Application

The final application is a Streamlit app.

From the project root:

```powershell
streamlit run src/app.py
```

Streamlit will display a local URL.

Open the URL in your browser.

---

# 📤 Using the Runtime Upload Mode

## Step 1 — Open the Knowledge Base panel

The application keeps the Knowledge Base/upload controls in the left sidebar.

If the sidebar is collapsed, click the:

```text
»
```

arrow in the top-left corner.

## Step 2 — Upload PDFs

You can upload multiple PDF files in one batch.

Example:

```text
research_paper_1.pdf
research_paper_2.pdf
employee_handbook.pdf
leave_policy.pdf
student_handbook.pdf
attendance_policy.pdf
mysql_manual.pdf
technical_guide.pdf
```

## Step 3 — Process

Click:

```text
Process Documents
```

The application:

```text
Reads PDFs
 ↓
Detects domain
 ↓
Chunks text
 ↓
Generates embeddings
 ↓
Builds runtime ChromaDB
```

## Step 4 — Ask Questions

Choose:

```text
All
```

or a specific domain:

```text
Research
Company
College
Technical
```

Then ask your question.

---

# 🧪 Example Questions

## College

```text
What is the minimum attendance requirement?
```

## Research

```text
What is the main idea behind the Transformer architecture?
```

## Company

```text
What are the employee leave policies?
```

## Technical

```text
What is the purpose of the Diagnostics and Recovery Toolset?
```

## Unknown / Out-of-domain

```text
What is the current temperature on Mars?
```

The last type of question should result in a safe refusal when the knowledge base does not contain the required evidence.

-------------------------------------------------------------------------------------------------------------------------------------------------

# 🧰 Main Technologies

| Technology | Purpose |
|---|---|
| Python | Core development language |
| LangChain | RAG/LLM integration |
| LangGraph | Stateful cyclic workflow |
| ChromaDB | Vector storage |
| Sentence Transformers | Local embeddings |
| Gemini | Answer generation, criticism, query rewriting |
| PyPDF | PDF text extraction |
| Streamlit | Web interface |
| Pydantic | Structured critic outputs |

---

# 📊 Example End-to-End Flow

Suppose a user uploads:

```text
karunya_student_handbook.pdf
burrell_student_handbook.pdf
```

and asks:

```text
What is the minimum attendance requirement?
```

The system performs:

```text
1. PDF Upload
        ↓
2. Page Extraction
        ↓
3. Domain = College
        ↓
4. Chunking
        ↓
5. Embeddings
        ↓
6. ChromaDB
        ↓
7. Semantic Retrieval
        ↓
8. Local Reranking
        ↓
9. Relevance Threshold
        ↓
10. Gemini Answer Generation
        ↓
11. Grounding Critic
        ↓
12. Grounded = True
        ↓
13. Final Answer + Sources
```

If the critic rejects the answer:

```text
Critic = False
      ↓
Query Rewriter
      ↓
New Query
      ↓
Re-retrieve
      ↓
Generate Again
      ↓
Critic
      ↓
Grounded = True
      ↓
Final Answer
```

---

# 📈 Evaluation Plan

For final academic evaluation, the system should be compared with a standard RAG baseline.

## Baseline RAG

```text
Retrieve
 ↓
Generate
```

## Self-Healing RAG

```text
Retrieve
 ↓
Generate
 ↓
Critic
 ↓
Rewrite / Retry
```

Recommended evaluation metrics:

- Answer correctness
- Groundedness
- Hallucination rate
- Retrieval relevance
- Successful recovery rate
- Average retry count
- Safe-refusal accuracy
- Response time

A comparison table can be built from actual test results:

| Metric | Standard RAG | Self-Healing RAG |
|---|---:|---:|
| Correct answers | Measured | Measured |
| Grounded answers | Measured | Measured |
| Hallucination rate | Measured | Measured |
| Successful recoveries | — | Measured |
| Average retries | — | Measured |
| Safe refusals | Measured | Measured |

Do not fill these values with invented numbers; they should come from the project's evaluation runs.

-------------------------------------------------------------------------------------------------------------------------------------------------

# ⚠️ Current Limitations

The system is designed for high reliability, but no RAG system can guarantee perfect accuracy for arbitrary documents.

Current limitations include:

1. PDF parsing quality depends on the source PDF.
2. Scanned/image-only PDFs may require OCR.
3. Automatic domain detection is heuristic.
4. Very ambiguous questions may require user clarification.
5. Larger document collections may require more advanced vector indexing/reranking.
6. Runtime knowledge bases are session-based and are not automatically persisted.
7. Gemini API availability/rate limits can affect generation.
8. The current embedding model is lightweight rather than a specialized domain embedding model.

-------------------------------------------------------------------------------------------------------------------------------------------------

# 🚀 Future Scope

Possible upgrades include:

- OCR support for scanned PDFs
- Persistent user-specific knowledge bases
- Document deletion/update
- Hybrid BM25 + vector retrieval
- Cross-encoder reranking
- Better domain classification
- Automatic ambiguity detection
- Conversation memory
- Streaming responses
- Advanced evaluation dashboard
- Retrieval analytics
- User authentication
- Cloud deployment
- Multi-user knowledge bases
- Database-backed document management
- Support for additional file types such as DOCX and TXT

-------------------------------------------------------------------------------------------------------------------------------------------------

# 📜 Project Workflow Summary

The final system can be summarized as:

```text
                  USER
                   │
                   ▼
            Upload PDF Files
                   │
                   ▼
           Document Processing
                   │
                   ▼
             Domain Detection
                   │
                   ▼
                Chunking
                   │
                   ▼
              Embeddings
                   │
                   ▼
               ChromaDB
                   │
                   ▼
              User Question
                   │
                   ▼
              Retrieval
                   │
                   ▼
               Reranking
                   │
                   ▼
          Relevance Threshold
                   │
              ┌────┴────┐
              │         │
          Relevant   Not Relevant
              │         │
              ▼         ▼
          Generation   Safe Refusal
              │
              ▼
            Critic
              │
        ┌─────┴─────┐
        │           │
     Grounded    Rejected
        │           │
        ▼           ▼
      Answer    Query Rewrite
                    │
                    ▼
                 Retrieve
                    │
                    └─────────────↺
```

-------------------------------------------------------------------------------------------------------------------------------------------------