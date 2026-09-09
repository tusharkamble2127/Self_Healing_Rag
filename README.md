# 🧠 Self-Healing RAG

A multi-domain Retrieval-Augmented Generation (RAG) system built with **LangChain, LangGraph, ChromaDB, Sentence Transformers, Gemini, and Streamlit**.

Unlike a traditional RAG pipeline, this system checks whether its generated answer is grounded in retrieved evidence. When retrieval or generation is weak, it can **rewrite the retrieval query, retry, and safely refuse** when sufficient evidence is unavailable.

The Streamlit application also supports **runtime PDF uploads**, allowing users to build a temporary knowledge base directly from the UI.

---

## 📌 Key Features

- Multi-domain PDF support: **Research, Company, College, Technical**
- Runtime PDF upload through Streamlit
- Semantic retrieval with ChromaDB
- Local reranking using semantic + lexical relevance
- Relevance threshold to reject weak retrieval
- Gemini-based answer generation
- Structured grounding critic
- Automatic query rewriting and retry using LangGraph
- Safe refusal for unsupported questions
- Source and page information in answers
- Optional domain filtering

---

## 🔄 How It Works

### Traditional RAG

```text
Question
   ↓
Retrieve
   ↓
Generate
   ↓
Answer
```

### Self-Healing RAG

```text
Question
   ↓
Retrieve
   ↓
Generate
   ↓
Critic
   ├── Grounded → Final Answer
   └── Rejected → Rewrite Query
                     ↓
                  Retrieve Again
                     ↓
                  Generate Again
                     ↓
                    Critic
```

After the retry limit is reached, the system returns a safe refusal instead of inventing an answer.

---

## 🏗️ Architecture

```text
                    ┌─────────────────┐
                    │  Streamlit UI   │
                    └────────┬────────┘
                             │
                       Upload PDFs
                             ↓
                    ┌─────────────────┐
                    │ PDF Processing  │
                    │ PyPDF + Splitter│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Domain Detection│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    Embeddings   │
                    │ all-MiniLM-L6-v2│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │    ChromaDB     │
                    └────────┬────────┘
                             ↓
                       User Question
                             ↓
                    ┌─────────────────┐
                    │   Retriever     │
                    │ + Reranking     │
                    │ + Threshold     │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Gemini Generator│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Grounding Critic│
                    └───────┬─┬───────┘
                            │ │
                     Accept │ │ Reject
                            │ └──────→ Query Rewriter
                            ↓                    │
                         Answer            Re-retrieve
```

---

## 🧩 Supported Domains

| Domain | Example Documents |
|---|---|
| 🔬 Research | Research papers, scientific papers |
| 🏢 Company | Employee handbooks, HR policies |
| 🎓 College | Student handbooks, attendance rules |
| 💻 Technical | Software and database documentation |

The user can search **all domains** or select one specific domain.

---

## 📚 Document Processing

Uploaded PDFs follow this pipeline:

```text
PDF
 ↓
Page Extraction
 ↓
Metadata + Domain Detection
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB
```

### Chunking

- `RecursiveCharacterTextSplitter`
- Chunk size: **1000 characters**
- Chunk overlap: **150 characters**

### Metadata

Each chunk stores metadata such as:

```python
{
    "source": "attendance_policy.pdf",
    "page": 20,
    "domain": "college",
    "document_type": "academic_policy",
    "chunk_id": 123
}
```

---

## 🔎 Retrieval and Reranking

The retriever first gets candidate chunks from ChromaDB and then reranks them.

```text
Query
 ↓
Candidate Search
 ↓
Semantic Score
 +
Lexical Overlap
 ↓
Combined Rerank Score
 ↓
Relevance Threshold
 ↓
Top Relevant Chunks
```

This reduces the chance of sending unrelated context to the LLM.

The embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings run locally, so no separate embedding API is required.

---

## 🤖 LLM Components

Gemini is used for three tasks:

### 1. Answer Generation

The model receives the question and retrieved context and is instructed to answer only from that context.

### 2. Grounding Critic

The critic checks:

```text
Question
+ Retrieved Context
+ Generated Answer
```

and returns structured output such as:

```json
{
  "grounded": true,
  "confidence": 0.95,
  "reason": "The answer is supported by the retrieved context."
}
```

### 3. Query Rewriting

When an answer is rejected, the system creates a more retrieval-friendly query without changing the original user intent.

Example:

```text
Original:
What is the minimum attendance requirement?

Rewritten:
minimum attendance requirement percentage eligibility policy
write examinations
```

---

## 🔁 LangGraph Workflow

The self-healing workflow is implemented as a cyclic LangGraph:

```text
START
  ↓
retrieve
  ↓
generate
  ↓
critic
  ├── success → END
  ├── rewrite → retrieve
  └── failure → END
```

The graph state tracks values such as:

```python
{
    "question": ...,
    "current_query": ...,
    "documents": ...,
    "answer": ...,
    "grounded": ...,
    "confidence": ...,
    "retry_count": ...,
    "final_response": ...,
    "domain": ...
}
```

The original question is preserved; only the retrieval query is rewritten.

### Retry Limit

```text
MAX_RETRIES = 2
```

So the system can make up to **3 total attempts**, including the first attempt.

---

## 🧪 Self-Healing Verification

The recovery loop was tested using a controlled development test:

```text
Incorrect first answer
        ↓
Critic rejects it
        ↓
Query rewritten
        ↓
Better evidence retrieved
        ↓
Correct answer generated
        ↓
Critic accepts it
```

The intentional test hook was removed from the production workflow after verification.

---

## 🛡️ Safe Refusal

The system is designed not to answer using unrelated evidence.

For an unsupported question such as:

```text
What is the current temperature on Mars?
```

weak retrieval can be rejected before generation, resulting in:

```text
I don't have enough information in the available documents
to provide a reliable answer.
```

---

## 🖥️ Streamlit Application

The UI supports:

- Multiple PDF uploads
- Mixed-domain documents
- Automatic domain detection
- Knowledge-base processing
- Domain selection
- Natural-language questions
- Verified answers
- Retry count
- Grounded status
- Confidence score
- Source/page information

The internal LangGraph trace remains available in the backend for debugging and evaluation.

---

## 📁 Project Structure

```text
Self_Healing_Rag/
│
├── src/
│   ├── app.py
│   ├── config.py
│   ├── critic.py
│   ├── generator.py
│   ├── graph_rag.py
│   ├── ingest.py
│   ├── main.py
│   ├── retriever.py
│   ├── rewriter.py
│   ├── runtime_kb.py
│   ├── self_healing_rag.py
│   └── vector_store.py
│
├── data/
│   └── raw/
│       ├── college/
│       ├── company/
│       ├── research/
│       └── technical/
│
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Create and activate virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
python -m pip install -U pip
pip install -r requirements.txt
```

### 3. Configure Gemini API key

Create `.env`:

```env
GEMINI_API_KEY=your_key_here
```

Do not commit `.env` to GitHub.

---

## ▶️ Run the Application

From the project root:

```powershell
streamlit run src/app.py
```

Open the local Streamlit URL shown in the terminal.

---

## 📤 Using the Runtime Knowledge Base

1. Open the left sidebar.
2. Upload one or more PDFs.
3. Click **Process Documents**.
4. Select **All** or a specific domain.
5. Enter a question.
6. View the grounded answer and sources.

The runtime knowledge base is temporary and belongs to the current application session.

---

## 🧪 Example Questions

**College**
```text
What is the minimum attendance requirement?
```

**Research**
```text
What is the main idea behind the Transformer architecture?
```

**Company**
```text
What are the employee leave policies?
```

**Technical**
```text
What is the purpose of the Diagnostics and Recovery Toolset?
```

**Out-of-domain**
```text
What is the current temperature on Mars?
```

---

## 🧰 Technologies

| Technology | Purpose |
|---|---|
| Python | Core development |
| LangChain | RAG and LLM integration |
| LangGraph | Stateful self-healing workflow |
| ChromaDB | Vector store |
| Sentence Transformers | Local embeddings |
| Gemini | Generation, critic, query rewriting |
| PyPDF | PDF extraction |
| Streamlit | User interface |
| Pydantic | Structured outputs |

---

## 📊 Evaluation

For academic evaluation, compare:

### Standard RAG

```text
Retrieve → Generate
```

### Self-Healing RAG

```text
Retrieve → Generate → Critic → Rewrite/Retry
```

Useful metrics:

- Answer correctness
- Groundedness
- Hallucination rate
- Retrieval relevance
- Successful recovery rate
- Average retry count
- Safe-refusal accuracy
- Response time

Use actual experiment results rather than estimated values.

---

## ⚠️ Limitations

- PDF quality affects extraction quality.
- Scanned PDFs may require OCR.
- Domain detection is heuristic.
- Ambiguous questions may still be difficult.
- Larger collections may need stronger retrieval/reranking.
- Runtime knowledge bases are session-based.
- Gemini availability and rate limits may affect generation.
- `all-MiniLM-L6-v2` is a lightweight embedding model.

---

## 🚀 Future Scope

- OCR for scanned PDFs
- Hybrid BM25 + vector retrieval
- Cross-encoder reranking
- Better domain classification
- Persistent user knowledge bases
- Conversation memory
- Streaming responses
- Evaluation dashboard
- Authentication and cloud deployment
- DOCX/TXT support

---

## 📌 Project Summary

```text
Upload PDFs
    ↓
Process + Detect Domain
    ↓
Chunk + Embed
    ↓
Store in ChromaDB
    ↓
Retrieve + Rerank
    ↓
Generate with Gemini
    ↓
Critic Checks Grounding
    ├── Accept → Final Answer
    └── Reject → Rewrite → Retry
```

The main goal is to make RAG more reliable by combining **retrieval, generation, evaluation, recovery, and safe refusal** in one LangGraph-based workflow.
