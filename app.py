# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Solution Assistant — Streamlit UI
# Features:
#   • Multi-session chat windows (create / rename / delete / switch)
#   • Per-chat document source filter
#   • Multi-document summarization with style selector  ← original
#   • Presales report generation (5 report types)       ← new
#   • Document upload for ALL file types (PDF, DOCX, TXT, MD,
#     CSV, XLSX, PPTX, HTML, PNG, JPG, JPEG, TIFF, BMP) ← new
#   • Sidebar document stats and LLM status
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import tempfile
import logging

import streamlit as st

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Solution Assistant",
    page_icon="📚",
    layout="wide",
)

# ─────────────────────────────────────────────
# Module path
# ─────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.rag_pipeline import RAGPipeline
from modules.report_pipeline import ReportPipeline
from modules.chunker import DocumentChunker
from modules.embedder import EmbeddingEngine
from modules.chat_manager import ChatManager
from modules.summarizer import DocumentSummarizer
import config


# ─────────────────────────────────────────────
# Helper — HTML report builder
# ─────────────────────────────────────────────

def _build_html_report(rr) -> str:
    """Wrap the markdown report in clean standalone HTML for download."""
    import re

    def md_to_html(markdown: str) -> str:
        lines = markdown.split("\n")
        html_lines = []
        in_list = False

        for line in lines:
            if line.startswith("### "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("## "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("# "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("- ") or line.startswith("* "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                item = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line[2:])
                html_lines.append(f"<li>{item}</li>")
            elif line.startswith("|"):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                cells = "".join(
                    f"<td>{c.strip()}</td>"
                    for c in line.strip("|").split("|")
                )
                html_lines.append(f"<tr>{cells}</tr>")
            elif line.startswith("---"):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append("<hr>")
            elif not line.strip():
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append("<br>")
            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
                line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)
                html_lines.append(f"<p>{line}</p>")

        if in_list:
            html_lines.append("</ul>")
        return "\n".join(html_lines)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{rr.report_type_label}</title>
  <style>
    body {{ font-family: Georgia, serif; max-width: 900px; margin: 40px auto;
            padding: 0 20px; line-height: 1.7; color: #1a1a2e; }}
    h1, h2, h3 {{ color: #16213e; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
    th {{ background: #f0f4f8; }}
    code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }}
    blockquote {{ border-left: 4px solid #4a90d9; margin: 0;
                  padding-left: 16px; color: #555; }}
    hr {{ border: none; border-top: 1px solid #ddd; margin: 24px 0; }}
  </style>
</head>
<body>
<h1>{rr.report_type_label}</h1>
<p><em>Sources: {', '.join(rr.source_documents)} |
   Generated in {rr.elapsed_seconds}s | {rr.word_count} words</em></p>
<hr>
{md_to_html(rr.markdown)}
</body>
</html>"""

# ─────────────────────────────────────────────
# Cached pipelines
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading RAG Pipeline...")
def get_pipeline():
    try:
        return RAGPipeline(), None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner="Loading Report Pipeline...")
def get_report_pipeline(_rag):
    """
    ReportPipeline shares the vector store and LLM from RAGPipeline.
    Underscore prefix on _rag prevents Streamlit from hashing the object.
    """
    try:
        return ReportPipeline(
            vector_store=_rag.vector_store,
            llm_handler=_rag.llm,
        ), None
    except Exception as e:
        return None, str(e)

pipeline, pipeline_error = get_pipeline()
report_pipeline, report_pipeline_error = (
    get_report_pipeline(pipeline) if pipeline else (None, "RAG pipeline failed")
)

# ─────────────────────────────────────────────
# Chat manager
# ─────────────────────────────────────────────

cm = ChatManager()
cm.ensure_default_session()

# ─────────────────────────────────────────────
# One-time session_state keys
# ─────────────────────────────────────────────

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

if "rename_target" not in st.session_state:
    st.session_state.rename_target = None

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"

if "last_report" not in st.session_state:
    st.session_state.last_report = None

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    .chat-tab-active {
        background: #1e6ef4;
        color: white !important;
        border-radius: 6px;
        padding: 2px 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════

with st.sidebar:

    st.title("📚 Solution Assistant")

    # ── LLM status ───────────────────────────

    st.subheader("🤖 LLM Status")
    if pipeline_error:
        st.error(pipeline_error)
    elif pipeline:
        llm_status = pipeline.llm_status()
        if llm_status["available"]:
            st.success(f"Online: {llm_status['model']}")
        else:
            st.error(f"Offline: {llm_status['model']}")
            st.code(f"ollama pull {llm_status['model']}")

    # ── Vector-store stats ────────────────────

    if pipeline:
        col_a, col_b = st.columns(2)
        col_a.metric("Chunks", pipeline.document_count())
        col_b.metric("Documents", len(pipeline.list_documents()))

    st.divider()

    # ── Chunk settings ────────────────────────

    with st.expander("✂️ Chunk Settings", expanded=False):
        chunk_size = st.slider("Chunk Size", 200, 2000, 500, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 50, 20)
        model_name = st.text_input("Embedding Model", value="all-MiniLM-L6-v2")

        # ── Chunk size warnings ───────────────────
        if chunk_size <= 300:
            st.warning(
                "⚠️ Chunk size is very small (< 300). "
                "Chunks may lack enough context for accurate retrieval."
            )
        elif chunk_size < 500:
            st.info(
                "ℹ️ Chunk size is below the recommended default (500). "
                "Good for short, precise documents."
            )
        elif chunk_size >= 1200:
            st.warning(
                "⚠️ Chunk size is very large (> 1200). "
                "This may reduce retrieval precision and slow embedding."
            )
        elif chunk_size >= 800:
            st.info(
                "ℹ️ Chunk size is above the recommended default (500). "
                "Good for long-form documents where broad context helps."
            )

        # ── Chunk overlap warnings ────────────────
        if chunk_overlap >= chunk_size:
            st.error(
                "❌ Chunk overlap must be smaller than chunk size. "
                "Reduce overlap or increase chunk size."
            )
        elif chunk_overlap > chunk_size * 0.4:
            st.warning(
                f"⚠️ Overlap ({chunk_overlap}) is more than 40% of chunk size ({chunk_size}). "
                "This creates heavy redundancy and increases storage."
            )
        elif chunk_overlap == 0:
            st.info(
                "ℹ️ Zero overlap means no context is shared between chunks. "
                "Sentences at chunk boundaries may lose meaning."
            )

    # ── Upload ────────────────────────────────
    # Now accepts all supported file types, not just PDF/DOCX

    st.subheader("📂 Upload Documents")

    accepted_types = [ext.lstrip(".") for ext in config.SUPPORTED_EXTENSIONS]

    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT, MD, CSV, XLSX, PPTX, HTML, PNG, JPG, TIFF, BMP",
        type=accepted_types,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and pipeline:
        for uf in uploaded_files:
            if uf.name not in st.session_state.ingested_files:
                with st.spinner(f"Ingesting {uf.name}..."):
                    ext = os.path.splitext(uf.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(uf.read())
                        temp_path = tmp.name
                    try:
                        pipeline.chunker = DocumentChunker(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        pipeline.embedder = EmbeddingEngine(model_name=model_name)
                        pipeline.retriever.embedding_engine = pipeline.embedder

                        result = pipeline.ingest(temp_path)
                        if result.success:
                            st.success(
                                f"✅ {uf.name} — "
                                f"{result.chunks_created} chunks ({result.elapsed_seconds}s)"
                            )
                            st.session_state.ingested_files.add(uf.name)
                        else:
                            st.error(result.error)
                    except Exception as e:
                        st.error(f"Ingestion error: {e}")
                    finally:
                        os.unlink(temp_path)

    # ── Indexed documents ─────────────────────

    if pipeline:
        docs = pipeline.list_documents()
        if docs:
            st.subheader("📄 Indexed Documents")
            for doc in docs:
                col1, col2 = st.columns([5, 1])
                col1.info(doc, icon="📄")
                if col2.button("❌", key=f"del_{doc}"):
                    pipeline.delete_document(doc)
                    st.session_state.ingested_files.discard(doc)
                    st.rerun()

    st.divider()

    # ── Chat sessions panel ───────────────────

    st.subheader("💬 Chat Sessions")

    if st.button("➕ New Chat", use_container_width=True):
        cm.new_session()
        st.session_state.active_tab = "chat"
        st.rerun()

    session_list = cm.session_list()
    active_id = cm.active_id()

    for sess in session_list:
        sid = sess["id"]
        is_active = sid == active_id
        label = ("▶ " if is_active else "   ") + sess["name"]
        badge = f"  ({sess['message_count']} msgs)" if sess["message_count"] else ""

        col_s, col_r, col_d = st.columns([6, 1, 1])

        if col_s.button(
            label + badge,
            key=f"switch_{sid}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            cm.set_active(sid)
            st.session_state.active_tab = "chat"
            st.rerun()

        if col_r.button("✏️", key=f"ren_{sid}"):
            st.session_state.rename_target = sid
            st.rerun()

        if col_d.button("🗑", key=f"del_sess_{sid}"):
            cm.delete_session(sid)
            st.rerun()

    # Rename input
    if st.session_state.rename_target:
        rt = st.session_state.rename_target
        current_name = cm.all_sessions().get(rt, {}).get("name", "")
        new_name = st.text_input(
            f"Rename '{current_name}'",
            value=current_name,
            key="rename_input",
        )
        c1, c2 = st.columns(2)
        if c1.button("Save", key="rename_save"):
            cm.rename_session(rt, new_name)
            st.session_state.rename_target = None
            st.rerun()
        if c2.button("Cancel", key="rename_cancel"):
            st.session_state.rename_target = None
            st.rerun()

# ═════════════════════════════════════════════
#  MAIN AREA — Three tabs
# ═════════════════════════════════════════════

active_session = cm.active_session()
session_name = active_session["name"] if active_session else "Chat"

st.header(f"💬 {session_name}")

tab_chat, tab_summary, tab_report = st.tabs([
    "🗨️ Chat",
    "📝 Summarize Documents",
    "📊 Generate Report",
])

# ─────────────────────────────────────────────
#  TAB 1 — Chat  (original, unchanged)
# ─────────────────────────────────────────────

with tab_chat:

    source_filter = None
    if pipeline:
        docs = pipeline.list_documents()
        if docs:
            pinned = cm.pinned_sources()
            default_filter = pinned[0] if pinned else "All Documents"
            filter_choice = st.selectbox(
                "🔍 Search within",
                ["All Documents"] + docs,
                index=(docs.index(default_filter) + 1) if default_filter in docs else 0,
                key=f"src_filter_{active_id}",
            )
            if filter_choice != "All Documents":
                source_filter = filter_choice

    if st.button("🗑 Clear Chat", key="clear_chat"):
        cm.clear_session()
        st.rerun()

    for msg in cm.active_messages():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📎 Sources ({len(msg['citations'])})"):
                    for cit in msg["citations"]:
                        st.markdown(
                            f"**📄 {cit['source']}** — "
                            f"Page {cit['page']}, Chunk {cit['chunk']}, "
                            f"Similarity {cit['similarity']:.2f}\n\n"
                            f"> {cit['excerpt']}"
                        )

    if pipeline and pipeline.document_count() == 0:
        st.info("Upload documents in the sidebar to begin.")

    disabled = pipeline is None or (pipeline and pipeline.document_count() == 0)

    if user_input := st.chat_input(
        "Ask a question about your documents...",
        disabled=disabled,
        key=f"chat_input_{active_id}",
    ):
        cm.add_message(role="user", content=user_input)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Searching documents and generating answer..."):
            response = pipeline.query(user_input, source_filter=source_filter)

        cm.add_message(
            role="assistant",
            content=response.answer,
            citations=response.citations,
        )

        with st.chat_message("assistant"):
            st.markdown(response.answer)
            st.caption(f"⏱ {response.elapsed_seconds}s")
            if response.citations:
                with st.expander(f"📎 Sources ({len(response.citations)})"):
                    for cit in response.citations:
                        st.markdown(
                            f"**📄 {cit['source']}** — "
                            f"Page {cit['page']}, Chunk {cit['chunk']}, "
                            f"Similarity {cit['similarity']:.2f}\n\n"
                            f"> {cit['excerpt']}"
                        )

        st.rerun()

# ─────────────────────────────────────────────
#  TAB 2 — Summarize Documents  (original, unchanged)
# ─────────────────────────────────────────────

with tab_summary:

    st.subheader("📝 Multi-Document Summarizer")
    st.caption(
        "Generate AI summaries for one or more ingested documents. "
        "Choose a style and select which documents to include."
    )

    if pipeline is None or pipeline_error:
        st.error("Pipeline not available. Check the sidebar for errors.")

    elif pipeline.document_count() == 0:
        st.info("No documents ingested yet. Upload files in the sidebar first.")

    else:
        docs = pipeline.list_documents()

        col_l, col_r = st.columns([2, 1])

        with col_l:
            selected_docs = st.multiselect(
                "Documents to summarize",
                options=docs,
                default=docs,
                placeholder="Select one or more documents...",
            )

        with col_r:
            style = st.selectbox(
                "Summary style",
                options=["concise", "detailed", "bullet"],
                format_func=lambda s: {
                    "concise": "🗜️ Concise (3-5 sentences)",
                    "detailed": "📋 Detailed (structured)",
                    "bullet": "• Bullet Points",
                }[s],
            )

        generate_combined = st.checkbox(
            "Generate cross-document synthesis (when multiple docs selected)",
            value=True,
        )

        if not selected_docs:
            st.warning("Select at least one document to summarize.")
        else:
            if st.button(
                f"⚡ Generate Summary{'ies' if len(selected_docs) > 1 else ''}",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner(
                    f"Summarizing {len(selected_docs)} document(s) — this may take a minute..."
                ):
                    result = pipeline.summarize(
                        sources=selected_docs,
                        style=style,
                        generate_combined=generate_combined,
                    )

                st.success(
                    f"Done! Summarized {result.total_documents} document(s) "
                    f"in {result.total_elapsed_seconds}s."
                )

                st.divider()
                st.subheader("📄 Individual Document Summaries")

                for ds in result.document_summaries:
                    icon = "✅" if ds.success else "❌"
                    with st.expander(
                        f"{icon} {ds.source}  ({ds.chunk_count} chunks, {ds.elapsed_seconds}s)",
                        expanded=True,
                    ):
                        if ds.success:
                            st.markdown(ds.summary)
                            if st.button(
                                "💬 Send to current chat",
                                key=f"to_chat_{ds.source}",
                            ):
                                msg = (
                                    f"**Summary of {ds.source}** "
                                    f"*(style: {ds.style})*\n\n{ds.summary}"
                                )
                                cm.add_message(role="assistant", content=msg)
                                st.session_state.active_tab = "chat"
                                st.toast(f"Summary sent to '{cm.active_session()['name']}'!")
                        else:
                            st.error(ds.error)

                if result.combined_summary:
                    st.divider()
                    st.subheader("🔗 Cross-Document Synthesis")
                    st.markdown(result.combined_summary)

                    if st.button("💬 Send synthesis to current chat", key="send_combined"):
                        msg = (
                            f"**Cross-Document Synthesis** "
                            f"*(style: {style}, docs: {', '.join(selected_docs)})*"
                            f"\n\n{result.combined_summary}"
                        )
                        cm.add_message(role="assistant", content=msg)
                        st.toast(f"Synthesis sent to '{cm.active_session()['name']}'!")

# ─────────────────────────────────────────────
#  TAB 3 — Generate Report
# ─────────────────────────────────────────────

with tab_report:

    st.subheader("📊 Presales Report Generator")
    st.caption(
        "Describe what you want to extract and the report will be built from "
        "your documents. Use a template or write your own instruction."
    )

    if pipeline is None or pipeline_error:
        st.error("Pipeline not available. Check the sidebar for errors.")

    elif report_pipeline is None:
        st.error(f"Report pipeline failed: {report_pipeline_error}")

    elif pipeline.document_count() == 0:
        st.info("No documents ingested yet. Upload files in the sidebar first.")

    else:
        docs = pipeline.list_documents()

        # ── Natural language request ──────────

        st.subheader("💬 What do you want to extract?")

        # Quick-fill example prompts
        EXAMPLE_PROMPTS = {
            "": "",
            "📋 Executive Summary":
                "Extract a concise executive summary including key highlights, "
                "critical data points, and recommended next steps for the presales team.",
            "🏆 Competitive Analysis":
                "Identify all competitors, alternative solutions, and market positioning "
                "mentioned in the documents. List strengths, weaknesses, and our differentiators.",
            "📌 Requirements Extraction":
                "Extract all functional and non-functional requirements. Label each as "
                "REQ-F-XXX or REQ-NF-XXX with priority (High / Medium / Low) and source.",
            "⚠️ Risk & Gap Assessment":
                "Identify all risks, blockers, compliance requirements, and capability gaps. "
                "Present as a table with likelihood, impact, and mitigation for each.",
            "📝 Proposal Outline":
                "Create a proposal outline including value proposition, solution overview, "
                "delivery approach, pricing signals, and recommended next steps.",
        }

        selected_example = st.selectbox(
            "Start from a template (or write your own below)",
            options=list(EXAMPLE_PROMPTS.keys()),
            format_func=lambda x: x if x else "✍️ Write my own...",
            key="report_example_select",
        )

        # Pre-fill the text area if a template was chosen
        prefill = EXAMPLE_PROMPTS.get(selected_example, "")

        user_report_request = st.text_area(
            "Your report instruction",
            value=prefill,
            placeholder=(
                "E.g. 'Extract all pricing information and commercial terms mentioned "
                "in the documents and present them in a structured table.' \n\n"
                "Or: 'Identify all technical integration requirements and flag any "
                "that involve third-party APIs or legacy systems.'"
            ),
            height=130,
            key="report_user_request",
        )

        st.divider()

        # ── Supporting options ────────────────

        col_docs, col_audience = st.columns(2)

        with col_docs:
            report_selected_docs = st.multiselect(
                "📄 Source Documents",
                docs,
                default=docs,
                help="Select which indexed documents to include in the report.",
                key="report_doc_select",
            )

        with col_audience:
            audience = st.text_input(
                "👤 Target Audience",
                value=config.REPORT_DEFAULT_AUDIENCE,
                help="E.g. 'VP of Sales', 'CTO', 'procurement committee'",
            )

        with st.expander("🔧 Advanced Settings", expanded=False):
            max_context = st.slider(
                "Max Context Characters",
                min_value=2000,
                max_value=16000,
                value=config.REPORT_MAX_CONTEXT_CHARS,
                step=500,
                help="More context = more complete analysis but slower generation.",
            )

        st.divider()

        # ── Validation & Generate button ──────

        request_ready = bool(user_report_request.strip()) and bool(report_selected_docs)

        if not user_report_request.strip():
            st.warning("✏️ Describe what you want to extract or choose a template above.")
        elif not report_selected_docs:
            st.warning("📄 Select at least one document.")
        else:
            st.info(
                f"Will analyse **{len(report_selected_docs)}** document(s) "
                f"for **{audience}**."
            )

        generate_clicked = st.button(
            "🚀 Generate Report",
            type="primary",
            disabled=not request_ready,
            key="gen_report_btn",
        )

        # ── Generation ────────────────────────

        if generate_clicked and request_ready:
            with st.spinner("Analysing documents and generating report... this may take 30–90 seconds."):
                report_result = report_pipeline.generate_report(
                    source_documents=report_selected_docs,
                    report_type="custom",          # always custom — driven by user text
                    custom_instruction=user_report_request.strip(),
                    audience=audience,
                    max_context_chars=max_context,
                )
            st.session_state.last_report = report_result

        # ── Display Report ────────────────────

        if st.session_state.last_report:
            rr = st.session_state.last_report

            st.divider()

            if not rr.success:
                st.error(f"❌ Report generation failed: {rr.error}")

            else:
                # Metrics row
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Report Type", rr.report_type_label)
                col_m2.metric("Word Count", f"{rr.word_count:,}")
                col_m3.metric("Generation Time", f"{rr.elapsed_seconds}s")

                # Sources used
                with st.expander(
                    f"📎 Sources Used ({len(rr.source_documents)})",
                    expanded=False,
                ):
                    for doc in rr.source_documents:
                        st.markdown(f"- 📄 `{doc}`")
                    meta = rr.metadata
                    st.caption(
                        f"Chunks analysed: {meta.get('chunks_used', '?')} | "
                        f"Context chars: {meta.get('context_chars', '?'):,} | "
                        f"Audience: {meta.get('audience', '?')}"
                    )

                st.divider()

                # Report content
                st.subheader(f"📋 {rr.report_type_label}")
                st.markdown(rr.markdown)

                st.divider()

                # Send to chat
                if st.button("💬 Send report to current chat", key="report_to_chat"):
                    msg = (
                        f"**{rr.report_type_label}** "
                        f"*(docs: {', '.join(rr.source_documents)})*"
                        f"\n\n{rr.markdown}"
                    )
                    cm.add_message(role="assistant", content=msg)
                    st.session_state.active_tab = "chat"
                    st.toast(f"Report sent to '{cm.active_session()['name']}'!")

                # Downloads
                col_dl1, col_dl2 = st.columns(2)

                with col_dl1:
                    st.download_button(
                        label="⬇️ Download as Markdown",
                        data=rr.markdown,
                        file_name=(
                            f"{rr.report_type}_"
                            f"{'_'.join(rr.source_documents[:2])}.md"
                        ).replace(" ", "_"),
                        mime="text/markdown",
                        key="dl_md",
                    )

                with col_dl2:
                    html_report = _build_html_report(rr)
                    st.download_button(
                        label="⬇️ Download as HTML",
                        data=html_report,
                        file_name=(
                            f"{rr.report_type}_"
                            f"{'_'.join(rr.source_documents[:2])}.html"
                        ).replace(" ", "_"),
                        mime="text/html",
                        key="dl_html",
                    )

