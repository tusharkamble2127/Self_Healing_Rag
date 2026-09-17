import streamlit as st

from graph_rag import (
    build_final_result,
    run_graph,
)

from retriever import (
    set_active_vector_store,
    clear_active_vector_store,
)

from runtime_kb import (
    build_uploaded_vector_store,
    load_uploaded_pdfs,
    summarize_chunks,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Self-Healing RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        /* ---------- Global typography ---------- */

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                         BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .main {
            padding-top: 1.2rem;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* ---------- Main title ---------- */

        .hero {
            padding: 1.7rem 2rem 1.5rem 2rem;
            border-radius: 22px;
            margin-bottom: 1.5rem;
            background:
                linear-gradient(
                    135deg,
                    rgba(84, 103, 255, 0.20),
                    rgba(138, 82, 255, 0.12),
                    rgba(0, 196, 255, 0.08)
                );
            border: 1px solid rgba(130, 140, 255, 0.25);
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.12);
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1.05;
            margin: 0;
        }

        .hero-subtitle {
            margin-top: 0.75rem;
            font-size: 1.15rem;
            line-height: 1.6;
            opacity: 0.78;
        }

        .hero-badge {
            display: inline-block;
            margin-top: 1rem;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            border: 1px solid rgba(130, 140, 255, 0.25);
            background: rgba(255, 255, 255, 0.06);
        }

        /* ---------- Section headings ---------- */

        .section-heading {
            font-size: 1.45rem;
            font-weight: 750;
            margin: 1.4rem 0 0.85rem 0;
            letter-spacing: -0.02em;
        }

        /* ---------- Cards ---------- */

        .card {
            padding: 1.15rem 1.25rem;
            border-radius: 18px;
            border: 1px solid rgba(128, 128, 128, 0.20);
            background: rgba(128, 128, 128, 0.035);
            box-shadow: 0 8px 26px rgba(0, 0, 0, 0.06);
            margin-bottom: 1rem;
        }

        .answer-card {
            padding: 1.45rem 1.5rem;
            border-radius: 20px;
            border: 1px solid rgba(100, 120, 255, 0.22);
            background: rgba(100, 120, 255, 0.055);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.08);
            margin: 0.6rem 0 1.2rem 0;
        }

        .answer-label {
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.65;
            margin-bottom: 0.55rem;
        }

        .answer-text {
            font-size: 1.15rem;
            line-height: 1.75;
        }

        /* ---------- Status cards ---------- */

        .status-card {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            border: 1px solid rgba(128, 128, 128, 0.16);
            background: rgba(128, 128, 128, 0.035);
            min-height: 98px;
        }

        .status-label {
            font-size: 0.82rem;
            font-weight: 700;
            opacity: 0.62;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .status-value {
            font-size: 1.55rem;
            font-weight: 800;
            margin-top: 0.3rem;
        }

        /* ---------- Source cards ---------- */

        .source-card {
            padding: 0.9rem 1rem;
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.15);
            background: rgba(128, 128, 128, 0.025);
            margin-bottom: 0.65rem;
            font-size: 1rem;
            line-height: 1.55;
        }

        .source-name {
            font-weight: 750;
        }

        /* ---------- Upload summary ---------- */

        .upload-summary {
            font-size: 1rem;
            line-height: 1.7;
        }

        .small-muted {
            opacity: 0.62;
            font-size: 0.93rem;
        }

        /* ---------- Buttons ---------- */

        .stButton > button {
            min-height: 2.9rem;
            font-size: 1rem;
            font-weight: 700;
            border-radius: 12px;
        }

        /* ---------- Inputs ---------- */

        textarea,
        input {
            font-size: 1.03rem !important;
        }

        /* ---------- Sidebar ---------- */

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.12);
        }

        [data-testid="stSidebar"] * {
            font-size: 0.98rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "domain_summary" not in st.session_state:
    st.session_state.domain_summary = {}

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# HELPERS
# ============================================================

def render_sources(sources):
    st.markdown(
        '<div class="section-heading">Sources</div>',
        unsafe_allow_html=True,
    )

    if not sources:
        st.info("No sources were returned.")
        return

    for source in sources:
        score = source.get("score")

        score_text = (
            f"{score:.3f}"
            if isinstance(score, (int, float))
            else "N/A"
        )

        st.markdown(
            f"""
            <div class="source-card">
                📄 <span class="source-name">{source['source']}</span>
                &nbsp;•&nbsp; Page {source['page']}
                &nbsp;•&nbsp; Domain: {source['domain']}
                &nbsp;•&nbsp; Retrieval score: {score_text}
            </div>
            """,
            unsafe_allow_html=True,
        )


def process_documents(uploaded_files):
    if not uploaded_files:
        raise ValueError(
            "Please upload at least one PDF."
        )

    documents = load_uploaded_pdfs(
        uploaded_files
    )

    if not documents:
        raise ValueError(
            "The uploaded PDFs could not be read."
        )

    vector_store, chunks = build_uploaded_vector_store(
        documents
    )

    st.session_state.vector_store = vector_store

    st.session_state.uploaded_files = [
        file.name
        for file in uploaded_files
    ]

    st.session_state.chunk_count = len(chunks)

    st.session_state.domain_summary = (
        summarize_chunks(chunks)
    )

    st.session_state.last_result = None

    set_active_vector_store(vector_store)


def ensure_uploaded_store_active():
    if st.session_state.vector_store is None:
        return False

    set_active_vector_store(
        st.session_state.vector_store
    )

    return True


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🧠 Self-Healing RAG</div>
        <div class="hero-subtitle">
            Ask questions across your uploaded documents with
            grounded retrieval, self-correction, and source-aware answers.
        </div>
        <div class="hero-badge">
            Retrieval • Generation • Critique • Self-Healing
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.info(
    "📚 To add your Knowledge Base, click the » arrow in the top-left "
    "corner to open the sidebar, then upload your PDF documents there, also select a domain."
)



# ============================================================
# SIDEBAR — KNOWLEDGE BASE
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="section-heading">Knowledge Base</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            "Upload research papers, company policies, "
            "college documents, technical documentation, "
            "or mixed-domain PDFs."
        ),
    )

    if uploaded_files:
        st.markdown(
            f'<div class="small-muted">'
            f'{len(uploaded_files)} file(s) selected'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander(
            "View selected files",
            expanded=False,
        ):
            for file in uploaded_files:
                st.write(f"📄 {file.name}")

        if st.button(
            "Process Documents",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(
                "Processing PDFs and building the knowledge base..."
            ):
                try:
                    process_documents(
                        uploaded_files
                    )

                    st.success(
                        "Knowledge base is ready."
                    )

                except Exception as exc:
                    st.error(
                        "Document processing failed."
                    )
                    st.exception(exc)

    if st.session_state.vector_store is not None:
        st.markdown(
            '<div class="section-heading">Current Knowledge Base</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Documents",
                len(
                    st.session_state.uploaded_files
                ),
            )

        with c2:
            st.metric(
                "Chunks",
                st.session_state.chunk_count,
            )

        st.markdown(
            '<div class="small-muted">Detected domains</div>',
            unsafe_allow_html=True,
        )

        for domain, count in (
            st.session_state.domain_summary.items()
        ):
            st.write(
                f"• {domain.title()}: {count} chunks"
            )

        st.divider()

        if st.button(
            "Clear Knowledge Base",
            use_container_width=True,
        ):
            st.session_state.vector_store = None
            st.session_state.uploaded_files = []
            st.session_state.chunk_count = 0
            st.session_state.domain_summary = {}
            st.session_state.last_result = None

            clear_active_vector_store()

            st.rerun()


# ============================================================
# MAIN — QUERY
# ============================================================

st.markdown(
    '<div class="section-heading">Ask your question</div>',
    unsafe_allow_html=True,
)

q_col1, q_col2 = st.columns(
    [4, 1.2],
    vertical_alignment="bottom",
)

with q_col1:
    question = st.text_area(
        "Question",
        placeholder=(
            "Example: What is the minimum attendance requirement?"
        ),
        height=130,
        label_visibility="collapsed",
    )

with q_col2:
    selected_domain = st.selectbox(
        "Domain",
        [
            "All",
            "Research",
            "Company",
            "College",
            "Technical",
        ],
        help=(
            "Use All to search the complete uploaded "
            "knowledge base."
        ),
    )

ask = st.button(
    "🔎  Ask Question",
    type="primary",
    use_container_width=True,
)


# ============================================================
# EXECUTE QUERY
# ============================================================
if ask:
    if not question.strip():
        st.warning(
            "Please enter a question."
        )
    elif not ensure_uploaded_store_active():
        st.warning(
            "Upload and process your PDFs first."
        )
    else:
        st.session_state.last_result = None

        with st.spinner(
            "Searching documents and verifying the answer..."
        ):
            try:
                result = run_graph(
                    question.strip(),
                    domain=selected_domain,
                )

                st.session_state.last_result = (
                    build_final_result(result)
                )

            except Exception as exc:
                error_message = str(exc)

                if "503" in error_message or "UNAVAILABLE" in error_message:
                    st.error(
                        "⚠️ Gemini is temporarily busy. "
                        "Please wait a few seconds and try again."
                    )
                elif "429" in error_message:
                    st.error(
                        "⚠️ Gemini API request limit reached. "
                        "Please wait and try again."
                    )
                else:
                    st.error(
                        "⚠️ The question could not be processed. "
                        "Please try again."
                    )


# ============================================================
# RESULT
# ============================================================

final_result = st.session_state.last_result

if final_result is not None:
    st.markdown(
        '<div class="section-heading">Answer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="answer-card">
            <div class="answer-label">Verified response</div>
            <div class="answer-text">
                {final_result["answer"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-label">Retries</div>
                <div class="status-value">
                    {final_result["retry_count"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        grounded_text = (
            "Yes"
            if final_result["grounded"]
            else "No"
        )

        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-label">Grounded</div>
                <div class="status-value">
                    {grounded_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        confidence = final_result.get(
            "confidence",
            0.0,
        )

        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-label">Confidence</div>
                <div class="status-value">
                    {confidence:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Deliberately no self-healing trace is shown in the UI.
    # The trace remains available internally in the backend for
    # evaluation/debugging and future analytics.

    render_sources(
        final_result["sources"]
    )
