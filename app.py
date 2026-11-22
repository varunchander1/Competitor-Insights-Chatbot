"""
app.py
------
Streamlit frontend for the Bank Annual Report RAG system.
Run with: PYTHONPATH=. streamlit run app/ui/app.py
"""

import sys
import os
import io
import json
import tempfile
import re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from rag_backend import (
    answer_query, RAGResponse, ConversationTurn,
    SYSTEM_PROMPT, parse_sections, linkify_footnotes,
    synthesise_table_from_answer, generate_devils_advocate,
)
from pipeline import (
    load_pdf, load_xls, chunk_documents, build_vector_store,
    load_vector_store, load_assets_for_pages,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INDEX_DIR     = os.path.join(os.path.dirname(__file__), '..', '..', 'faiss_index')
MANIFEST_PATH = os.path.join(INDEX_DIR, 'manifest.json')

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bank Report Analyst",
    page_icon="🏦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest() -> list[str]:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return []


def save_manifest(filenames: list[str]):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(filenames, f)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = load_manifest()

if "vector_store" not in st.session_state:
    if os.path.exists(INDEX_DIR):
        with st.spinner("Loading existing index..."):
            try:
                st.session_state.vector_store = load_vector_store(INDEX_DIR)
            except Exception as e:
                st.warning(f"Could not load existing index: {e}")
                st.session_state.vector_store = None
    else:
        st.session_state.vector_store = None

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = SYSTEM_PROMPT

if "synthesised_tables" not in st.session_state:
    st.session_state.synthesised_tables = {}

if "devils_advocates" not in st.session_state:
    st.session_state.devils_advocates = {}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_uploaded_files(uploaded_files):
    to_process = [f for f in uploaded_files if f.name not in st.session_state.ingested_files]
    if not to_process:
        return

    all_chunks = []
    failed     = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for uploaded_file in to_process:
            try:
                tmp_path = os.path.join(tmpdir, uploaded_file.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.read())

                ext = Path(uploaded_file.name).suffix.lower()
                if ext in (".xlsx", ".xls"):
                    pages = load_xls(tmp_path)
                else:
                    pages, _ = load_pdf(tmp_path)

                chunks = chunk_documents(pages)
                all_chunks.extend(chunks)
            except Exception as e:
                failed.append(uploaded_file.name)
                st.error(f"Failed to process {uploaded_file.name}: {e}")

        if all_chunks:
            try:
                if st.session_state.vector_store is not None:
                    from langchain_openai import OpenAIEmbeddings
                    from langchain_community.vectorstores import FAISS
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    new_store  = FAISS.from_documents(all_chunks, embeddings)
                    st.session_state.vector_store.merge_from(new_store)
                else:
                    tmp_index = os.path.join(tmpdir, "faiss_index")
                    st.session_state.vector_store = build_vector_store(all_chunks, tmp_index)

                st.session_state.vector_store.save_local(INDEX_DIR)
                newly_ingested = [f.name for f in to_process if f.name not in failed]
                st.session_state.ingested_files.extend(newly_ingested)
                save_manifest(st.session_state.ingested_files)

            except Exception as e:
                st.error(f"Failed to build vector store: {e}")


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

def build_pdf(msg: dict, msg_idx: int) -> bytes:
    """
    Build a PDF report from a single assistant message using reportlab.
    Sections are rendered linearly: insight → detail → data table →
    devil's advocate → sources/assumptions → citations.
    Returns the PDF as bytes for st.download_button.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    buffer   = io.BytesIO()
    doc      = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm,   bottomMargin=2*cm,
    )
    styles   = getSampleStyleSheet()
    story    = []

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=16, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"),
    )
    h2_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=12, spaceBefore=14, spaceAfter=4,
        textColor=colors.HexColor("#2c3e7a"),
        borderPad=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9.5, leading=14, spaceAfter=4,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=8, textColor=colors.grey, spaceAfter=6,
    )
    source_style = ParagraphStyle(
        "Source",
        parent=styles["Normal"],
        fontSize=8.5, leading=12, spaceAfter=3,
        leftIndent=12,
    )

    def clean(text: str) -> str:
        """Strip markdown and [^n] markers for PDF rendering."""
        text = re.sub(r'\[\^\d+\]', '', text)          # footnote markers
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # bold
        text = re.sub(r'\*(.*?)\*',   r'<i>\1</i>', text)    # italic
        text = re.sub(r'#{1,4}\s*',   '',            text)   # headings
        return text.strip()

    def add_section(title: str, text: str):
        if not text or not text.strip():
            return
        story.append(Paragraph(title, h2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Spacer(1, 4))
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 4))
                continue
            # Bullet lines
            if line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
                line = "• " + line[2:]
            story.append(Paragraph(clean(line), body_style))
        story.append(Spacer(1, 8))

    # ---- Title ----
    query = msg.get("query", "Bank Report Analysis")
    story.append(Paragraph("🏦 Bank Report Analyst", title_style))
    story.append(Paragraph(f"<i>Query: {query}</i>", caption_style))
    story.append(Paragraph(
        f"<i>Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}</i>",
        caption_style,
    ))
    story.append(Spacer(1, 12))

    sections = msg.get("sections") or parse_sections(msg.get("content", ""))

    # ---- Section 1: Insight ----
    add_section("💡 Final Condensed Insight", sections.get("insight", ""))

    # ---- Section 2: Supporting Detail ----
    add_section("📊 Supporting Detail & Rationale", sections.get("detail", ""))

    # ---- Section 3: Data Table ----
    table_data = st.session_state.synthesised_tables.get(str(msg_idx))
    if table_data and table_data.get("rows"):
        story.append(Paragraph("📋 Data Table", h2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Spacer(1, 4))
        story.append(Paragraph(table_data.get("caption", ""), caption_style))

        headers  = table_data.get("headers", [])
        rows     = table_data.get("rows",    [])
        tbl_data = [headers] + rows

        col_count = len(headers)
        col_width = (A4[0] - 4*cm) / col_count

        tbl = Table(tbl_data, colWidths=[col_width] * col_count, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#2c3e7a")),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0),  8),
            ("FONTSIZE",    (0, 1), (-1, -1), 7.5),
            ("BACKGROUND",  (0, 1), (-1, -1), colors.whitesmoke),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.whitesmoke, colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("VALIGN",      (0, 0), (-1, -1), "TOP"),
            ("PADDING",     (0, 0), (-1, -1), 4),
            ("WORDWRAP",    (0, 0), (-1, -1), True),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 12))

    # ---- Section 4: Devil's Advocate ----
    da_text = st.session_state.devils_advocates.get(str(msg_idx))
    if da_text:
        add_section("😈 Devil's Advocate", da_text)

    # ---- Section 5: Sources, Evidence & Assumptions ----
    add_section("📎 Sources, Evidence & Assumptions", sections.get("sources", ""))

    # ---- Section 6: Citations ----
    citations = msg.get("citations", [])
    if citations:
        story.append(Paragraph("📚 Retrieved Sources", h2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Spacer(1, 4))
        for c in citations:
            n        = c.get("index", "?")
            page_str = f"p.{c['page']}" if c.get("page") else "XLS data"
            story.append(Paragraph(
                f"<b>[{n}] {c['doc']}</b> — {page_str}",
                source_style,
            ))
            story.append(Paragraph(
                f"<i>{c.get('snippet', '')[:200]}...</i>",
                caption_style,
            ))
            story.append(Spacer(1, 4))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_with_footnote_links(text: str, msg_idx: int):
    linked = linkify_footnotes(text, msg_idx)
    st.markdown(linked, unsafe_allow_html=True)


def render_citations(citations: list[dict], msg_idx: int):
    if not citations:
        return
    with st.expander(f"📚 All retrieved sources ({len(citations)})", expanded=False):
        for c in citations:
            n        = c.get("index", "?")
            page_str = f"p.{c['page']}" if c.get("page") else "XLS data"
            anchor   = f"citation-{msg_idx}-{n}"
            st.markdown(f'<span id="{anchor}"></span>', unsafe_allow_html=True)
            st.markdown(f"**[^{n}] {c['doc']}** — {page_str}")
            st.caption(c["snippet"])
            if c != citations[-1]:
                st.divider()


def render_assets(assets: list[dict]):
    if not assets:
        return
    has_content = [a for a in assets if a.get("tables") or a.get("image_path")]
    if not has_content:
        return

    with st.expander(f"📊 Source pages & tables ({len(has_content)} pages)", expanded=False):
        for asset in has_content:
            doc      = asset.get("doc",    "")
            page     = asset.get("page",   "")
            tables   = asset.get("tables", [])
            img_path = asset.get("image_path")

            st.markdown(f"**{doc} — p.{page}**")
            col1, col2 = st.columns([1, 1])

            with col1:
                if tables:
                    import pandas as pd
                    for t_idx, table in enumerate(tables):
                        if not table:
                            continue
                        try:
                            header = table[0]
                            rows   = table[1:]
                            if header and all(isinstance(h, str) and h.strip() for h in header):
                                df = pd.DataFrame(rows, columns=header)
                            else:
                                df = pd.DataFrame(table)
                            df = df.fillna("").astype(str)
                            st.caption(f"Table {t_idx + 1}")
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        except Exception:
                            pass
                else:
                    st.caption("No tables extracted from this page.")

            with col2:
                if img_path and os.path.exists(img_path):
                    st.image(img_path, caption=f"{doc} p.{page}", use_container_width=True)
                else:
                    st.caption("No page image available.")

            st.divider()


def render_synthesised_table_box(msg_idx: int, query: str, answer: str):
    """Auto-generates on the rerun after the answer appears."""
    table_key   = str(msg_idx)
    pending_key = f"pending_table_{msg_idx}"

    with st.container(border=True):
        st.markdown("#### 📋 Data Table")

        if table_key in st.session_state.synthesised_tables:
            table = st.session_state.synthesised_tables[table_key]
            if table is None:
                st.caption("_No structured data found to tabulate for this answer._")
                if st.button("↺ Try again", key=f"retry_table_{msg_idx}", use_container_width=True):
                    del st.session_state.synthesised_tables[table_key]
                    st.rerun()
            else:
                import pandas as pd
                try:
                    caption = table.get("caption", "Extracted Data")
                    headers = table.get("headers", [])
                    rows    = table.get("rows",    [])
                    df      = pd.DataFrame(rows, columns=headers)
                    df      = df.fillna("Not reported").astype(str)
                    st.caption(caption)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.caption(f"_Could not render table: {e}_")
                if st.button("↺ Regenerate", key=f"regen_table_{msg_idx}", use_container_width=True):
                    del st.session_state.synthesised_tables[table_key]
                    st.rerun()

        elif st.session_state.get(pending_key):
            with st.spinner("Generating table..."):
                result = synthesise_table_from_answer(query, answer)
                st.session_state.synthesised_tables[table_key] = result
                st.session_state.pop(pending_key, None)
            st.rerun()
        else:
            st.caption("_Generating table..._")
            st.session_state[pending_key] = True
            st.rerun()


def render_devils_advocate_box(msg_idx: int, query: str, answer: str):
    """
    Devil's advocate box — auto-generates after the main answer.
    Uses a separate model to get genuinely different reasoning.
    """
    da_key      = str(msg_idx)
    pending_key = f"pending_da_{msg_idx}"

    with st.container(border=True):
        st.markdown("#### 😈 Devil's Advocate")
        st.caption("_Independent critical challenge from a separate model_")

        if da_key in st.session_state.devils_advocates:
            da_text = st.session_state.devils_advocates[da_key]
            if not da_text:
                st.caption("_Could not generate critique._")
                if st.button("↺ Retry critique", key=f"retry_da_{msg_idx}", use_container_width=True):
                    del st.session_state.devils_advocates[da_key]
                    st.rerun()
            else:
                st.markdown(da_text)
                if st.button("↺ Regenerate critique", key=f"regen_da_{msg_idx}", use_container_width=True):
                    del st.session_state.devils_advocates[da_key]
                    st.rerun()

        elif st.session_state.get(pending_key):
            with st.spinner("Generating critique..."):
                result = generate_devils_advocate(query, answer)
                st.session_state.devils_advocates[da_key] = result
                st.session_state.pop(pending_key, None)
            st.rerun()
        else:
            st.caption("_Generating critique..._")
            st.session_state[pending_key] = True
            st.rerun()


def render_response(msg: dict, msg_idx: int = 0):
    """Render an assistant message as structured boxes."""
    sections  = msg.get("sections") or parse_sections(msg.get("content", ""))
    followups = msg.get("followups", [])
    assets    = msg.get("assets",    [])
    citations = msg.get("citations", [])
    query     = msg.get("query",     "")
    answer    = msg.get("content",   "")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("#### 💡 Final Condensed Insight")
            if sections.get("insight"):
                render_with_footnote_links(sections["insight"], msg_idx)
            else:
                st.markdown("_No insight section found._")

        if sections.get("detail"):
            with st.container(border=True):
                st.markdown("#### 📊 Supporting Detail & Rationale")
                render_with_footnote_links(sections["detail"], msg_idx)

    with right:
        if sections.get("sources"):
            with st.container(border=True):
                st.markdown("#### 📎 Sources, Evidence & Assumptions")
                render_with_footnote_links(sections["sources"], msg_idx)

        if followups:
            with st.container(border=True):
                st.markdown("#### 🔁 Suggested Follow-up Questions")
                for q in followups:
                    if st.button(q, key=f"followup_{msg_idx}_{hash(q)}", use_container_width=True):
                        st.session_state["prefill_query"] = q

    # Full width — data table (auto-generated)
    render_synthesised_table_box(msg_idx, query, answer)

    # Full width — devil's advocate (auto-generated)
    render_devils_advocate_box(msg_idx, query, answer)

    # Full width — source pages and citations
    if assets:
        render_assets(assets)
    render_citations(citations, msg_idx)

    st.caption(f"_Answer drawn from {msg.get('chunks_used', '?')} document excerpts._")

    # PDF export button — full width, at the bottom of each response
    _render_pdf_export(msg, msg_idx)


def _render_pdf_export(msg: dict, msg_idx: int):
    """Render the PDF export button for a single response."""
    export_key = f"export_pdf_{msg_idx}"

    # Only show once table + DA are ready (or after a short wait)
    table_ready = str(msg_idx) in st.session_state.synthesised_tables
    da_ready    = str(msg_idx) in st.session_state.devils_advocates

    if table_ready and da_ready:
        try:
            pdf_bytes = build_pdf(msg, msg_idx)
            query     = msg.get("query", "analysis")
            filename  = f"bank_report_{re.sub(r'[^a-z0-9]', '_', query[:40].lower())}.pdf"
            st.download_button(
                label    = "⬇️ Export as PDF",
                data     = pdf_bytes,
                file_name= filename,
                mime     = "application/pdf",
                key      = export_key,
                use_container_width=True,
            )
        except Exception as e:
            st.caption(f"_PDF export unavailable: {e}_")
    else:
        st.caption("_PDF export available once table and critique are ready._")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🏦 Bank Report Analyst")
    st.markdown("---")

    st.markdown("### 📂 Upload Reports")
    uploaded_files = st.file_uploader(
        "Drop PDF or XLS annual reports here",
        type=["pdf", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        with st.spinner("Ingesting documents..."):
            ingest_uploaded_files(uploaded_files)

    st.markdown("### 📄 Ingested Reports")
    if st.session_state.ingested_files:
        for fname in st.session_state.ingested_files:
            st.success(f"✅ {fname}")
    else:
        st.info("No reports ingested yet. Upload PDFs above.")

    st.markdown("---")

    if st.session_state.ingested_files:
        st.markdown("**Example questions:**")
        example_qs = [
            "Compare NatWest and Lloyds' CET1 capital ratios",
            "What are the main risk factors cited by both banks?",
            "How did NIM change year-over-year for Lloyds?",
            "What are each bank's stated strategic priorities?",
            "Compare the cost-to-income ratios of both banks",
        ]
        for q in example_qs:
            if st.button(q, key=q, use_container_width=True):
                st.session_state["prefill_query"] = q

    st.markdown("---")

    with st.expander("⚙️ System Prompt", expanded=False):
        edited_prompt = st.text_area(
            "Edit analyst instructions",
            value=st.session_state.system_prompt,
            height=300,
            label_visibility="collapsed",
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply", use_container_width=True):
                st.session_state.system_prompt = edited_prompt
                st.success("Prompt updated!")
        with col2:
            if st.button("Reset", use_container_width=True):
                st.session_state.system_prompt = SYSTEM_PROMPT
                st.rerun()

    st.markdown("---")

    if st.button("🗑️ Clear index & start fresh", use_container_width=True):
        import shutil
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        st.session_state.vector_store       = None
        st.session_state.ingested_files     = []
        st.session_state.messages           = []
        st.session_state.history            = []
        st.session_state.synthesised_tables = {}
        st.session_state.devils_advocates   = {}
        st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🏦 Bank Annual Report Analyst")
st.caption("Upload annual reports in the sidebar, then ask anything about them.")

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_response(msg, msg_idx=i)
        else:
            st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

prefill = st.session_state.pop("prefill_query", "")
query   = st.chat_input(
    "Ask a question about the uploaded reports...",
    disabled=st.session_state.vector_store is None,
)

if prefill and not query:
    query = prefill

# ---------------------------------------------------------------------------
# Handle query
# ---------------------------------------------------------------------------

# app.py -> Handle query section

if query:
    # 1. Add User message to state immediately
    st.session_state.messages.append({"role": "user", "content": query})
    
    # 2. Trigger a rerun so the loop above renders the user's chat bubble
    st.rerun()

# This part handles the assistant's turn if the last message was from the user
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_query = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                result = answer_query(
                    query=last_query,
                    vector_store=st.session_state.vector_store,
                    history=st.session_state.history
                )
                
                assets = load_assets_for_pages(result.citations)
                
                msg = {
                    "role":        "assistant",
                    "content":     result.answer,
                    "sections":    result.sections,
                    "citations":   result.citations,
                    "followups":   result.followups,
                    "chunks_used": result.chunks_used,
                    "assets":      assets,
                    "query":       last_query,
                }

                # 3. Save to history and state
                st.session_state.history.append(
                    ConversationTurn(user=last_query, assistant=result.answer)
                )
                st.session_state.messages.append(msg)
                
                # 4. Final Rerun to render the structured assistant response
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption("Answers are generated from the ingested annual reports only. Always verify key figures independently.")