"""
rag_backend.py
--------------
RAG pipeline with:
  - Conversation memory (last N exchanges passed as context to the LLM)
  - Retrieval + re-ranking
  - On-demand table synthesis from answer text (called explicitly by UI)
  - Section parsing (splits response into 3 structured sections)
  - Follow-up question generation
  - Cited answer generation via OpenAI
  - Footnote link helpers
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# System prompt — loaded from file if available
# ---------------------------------------------------------------------------

_prompt_path = os.path.join(os.path.dirname(__file__), 'prompts/system_prompt.md')
with open(_prompt_path, 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOP_K        = 12
MAX_TOKENS   = 16000
MEMORY_TURNS = 3
MODEL        = "gpt-4o"

# Devil's advocate uses a different model to get genuinely different reasoning
DEVILS_ADVOCATE_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text:       str
    source_doc: str
    page:       Optional[int]   = None
    score:      Optional[float] = None


@dataclass
class RAGResponse:
    answer:            str
    sections:          dict       # {"insight": str, "detail": str, "sources": str}
    citations:         list[dict]
    chunks_used:       int
    followups:         list[str]  = field(default_factory=list)
    query:             str        = ""
    # synthesised_table removed from auto-pipeline — generated on demand via UI


@dataclass
class ConversationTurn:
    user:      str
    assistant: str


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_chunks(query: str, vector_store, top_k: int = TOP_K) -> list[Chunk]:
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return [
        Chunk(
            text=doc.page_content,
            source_doc=doc.metadata.get("source_doc", "Unknown Document"),
            page=doc.metadata.get("page", None),
            score=float(score),
        )
        for doc, score in results
    ]


def rerank_chunks(query: str, chunks: list[Chunk]) -> list[Chunk]:
    query_terms = set(query.lower().split())
    def overlap(chunk: Chunk) -> float:
        return len(query_terms & set(chunk.text.lower().split())) / max(len(query_terms), 1)
    return sorted(chunks, key=overlap, reverse=True)


# ---------------------------------------------------------------------------
# On-demand table synthesis
# ---------------------------------------------------------------------------

def synthesise_table_from_answer(query: str, answer: str) -> dict | None:
    """
    Generate a structured comparison table from the answer text itself.

    Uses the clean figures the LLM already reasoned over rather than noisy
    PDF extraction. Enforces no empty cells and full dual-bank coverage.

    Returns {"caption": str, "headers": [...], "rows": [[...]]} or None.
    """
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_completion_tokens=2000,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial data analyst specialising in UK banking. "
                        "Given a question and a detailed analyst answer, extract all quantitative "
                        "and qualitative data points into a clean structured comparison table.\n\n"
                        "STRICT RULES — follow every one:\n"
                        "1. NEVER leave a cell empty. If a figure for one bank is not explicitly "
                        "stated, write 'Not reported' or infer a directional note like '>industry avg'.\n"
                        "2. If both NatWest and another bank are discussed, ALWAYS produce side-by-side "
                        "columns — one column per bank. Do not put both banks in the same cell.\n"
                        "3. Every row MUST have a value in EVERY column — no blanks, no nulls.\n"
                        "4. Include both quantitative figures AND qualitative descriptors "
                        "(e.g. strategy, risk posture, competitive position).\n"
                        "5. Use GBP formatting: £bn for large figures, bps for margins, % for ratios.\n"
                        "6. Add a 'Delta / Interpretation' column summarising the difference or implication.\n"
                        "7. If the answer genuinely contains no structured data at all, return the "
                        "JSON string null (not an object).\n\n"
                        "Output ONLY valid JSON in exactly this format, no preamble or explanation:\n"
                        '{"caption": "Descriptive table title", '
                        '"headers": ["Metric", "NatWest", "Lloyds", "Delta / Interpretation"], '
                        '"rows": [["metric name", "natwest value", "lloyds value", "interpretation"]]}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"Analyst answer:\n{answer}\n\n"
                        "Extract every data point into the table. "
                        "For any metric where only one bank's figure appears in the answer, "
                        "fill the other bank's cell with 'Not reported in excerpts' rather than leaving it blank."
                    ),
                },
            ],
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        if raw.lower() == "null" or not raw:
            return None

        result = json.loads(raw)
        if not ("headers" in result and "rows" in result and result["rows"]):
            return None

        # Post-process: replace any remaining empty strings with "Not reported"
        cleaned_rows = []
        for row in result["rows"]:
            cleaned_rows.append([
                cell if (cell and str(cell).strip()) else "Not reported"
                for cell in row
            ])
        result["rows"] = cleaned_rows
        return result

    except Exception as e:
        print(f"  Table synthesis failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Devil's advocate critique
# ---------------------------------------------------------------------------

DEVILS_ADVOCATE_PROMPT = """You are a sceptical senior risk officer and contrarian analyst.
Your job is to critically challenge the analyst's answer provided to you.

You must:
1. Identify any assumptions that may not hold
2. Flag data gaps or where the evidence base is thin
3. Present alternative interpretations of the same data
4. Highlight risks or downsides that were understated or omitted
5. Question any forward-looking statements or projections

Be direct, specific, and constructive. Do NOT simply repeat what was said.
Format your response as concise bullet points — maximum 6 bullets.
Start each bullet with a bold label e.g. **Assumption risk:**, **Data gap:**, **Alternative view:**
"""

def generate_devils_advocate(query: str, answer: str) -> str:
    """
    Generate a devil's advocate critique of the analyst answer using a
    deliberately different model to get genuinely different reasoning.

    Returns a markdown string with bullet-point critique, or empty string on failure.
    """
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=DEVILS_ADVOCATE_MODEL,
            max_tokens=800,
            messages=[
                {"role": "system", "content": DEVILS_ADVOCATE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Original question: {query}\n\n"
                        f"Analyst answer:\n{answer}\n\n"
                        "Provide your critical challenge."
                    ),
                },
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Devil's advocate failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Footnote link helpers
# ---------------------------------------------------------------------------

def linkify_footnotes(text: str, msg_idx: int) -> str:
    """
    Convert [^n] markers in text into HTML superscript links that jump
    to the matching citation anchor in the 'All retrieved sources' expander.

    [^1]  →  <sup><a href="#citation-{msg_idx}-1">[1]</a></sup>
    """
    def replace(match):
        n = match.group(1)
        return (
            f'<sup><a href="#citation-{msg_idx}-{n}" '
            f'style="color:#4A90E2;text-decoration:none;font-weight:bold;">'
            f'[{n}]</a></sup>'
        )
    return re.sub(r'\[\^(\d+)\]', replace, text)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_context_block(chunks: list[Chunk]) -> str:
    lines = ["--- RETRIEVED EXCERPTS ---\n"]
    for i, chunk in enumerate(chunks, start=1):
        page_str = f", p.{chunk.page}" if chunk.page is not None else ""
        lines.append(f"[Excerpt {i}] Source: {chunk.source_doc}{page_str}")
        lines.append(chunk.text.strip())
        lines.append("")
    return "\n".join(lines)


def build_messages(
    query:   str,
    chunks:  list[Chunk],
    history: list[ConversationTurn],
) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history[-MEMORY_TURNS:]:
        messages.append({"role": "user",      "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})

    context = build_context_block(chunks)
    messages.append({
        "role": "user",
        "content": (
            f"{context}\n\n"
            f"--- QUESTION ---\n{query}\n\n"
            "Follow your instructions precisely. "
            "Use [^n] footnote markers referencing the excerpt numbers above for all citations."
        ),
    })
    return messages


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(
    query:   str,
    chunks:  list[Chunk],
    history: list[ConversationTurn],
) -> str:
    client   = OpenAI()
    messages = build_messages(query, chunks, history)

    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=MAX_TOKENS,
        messages=messages,
    )

    content = response.choices[0].message.content
    if not content:
        finish = response.choices[0].finish_reason
        raise ValueError(
            f"LLM returned an empty response (finish_reason='{finish}'). "
            "Try increasing MAX_TOKENS or reducing TOP_K."
        )
    return content


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

def parse_sections(answer: str) -> dict:
    sections = {"insight": "", "detail": "", "sources": ""}
    mapping  = {"1": "insight", "2": "detail", "3": "sources"}

    parts   = re.split(r'((?:#+\s*)?\d+\))', answer)
    current = None

    for part in parts:
        header = re.match(r'(?:#+\s*)?(\d+)\)', part.strip())
        if header:
            current = mapping.get(header.group(1))
        elif current:
            sections[current] += part

    sections = {k: v.strip() for k, v in sections.items()}

    if not any(sections.values()):
        sections["insight"] = answer

    return sections


# ---------------------------------------------------------------------------
# Follow-up question generation
# ---------------------------------------------------------------------------

def generate_followups(query: str, answer: str) -> list[str]:
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_completion_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial analyst assistant. "
                        "Given a question and answer about bank annual reports, "
                        "suggest exactly 3 concise follow-up questions the user might want to ask. "
                        "Output ONLY a JSON array of 3 strings, no preamble, no markdown fences. "
                        'Example: ["Question 1?", "Question 2?", "Question 3?"]'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nAnswer summary: {answer[:600]}",
                },
            ],
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  Follow-up generation failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def extract_citations(chunks: list[Chunk]) -> list[dict]:
    """
    Build citation list from retrieved chunks, numbered 1..n to match
    the [^n] markers used in the answer text.
    """
    seen, citations = set(), []
    for i, chunk in enumerate(chunks, start=1):
        key = (chunk.source_doc, chunk.page)
        if key not in seen:
            seen.add(key)
            citations.append({
                "index":   i,                              # matches [^n] in answer
                "doc":     chunk.source_doc,
                "page":    chunk.page,
                "snippet": chunk.text[:300].strip() + "...",
            })
    return citations


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer_query(
    query:   str,
    vector_store,
    history: list[ConversationTurn] = None,
) -> RAGResponse:
    """
    Full RAG pipeline: retrieve → rerank → generate → parse → follow-ups → citations.
    Table synthesis is NOT run here — it's triggered on-demand from the UI.
    """
    if history is None:
        history = []

    # 1. Retrieve
    chunks = retrieve_chunks(query, vector_store, top_k=TOP_K)

    if not chunks:
        return RAGResponse(
            answer="I could not find any relevant information in the ingested documents to answer your question.",
            sections={"insight": "No relevant information found.", "detail": "", "sources": ""},
            citations=[],
            chunks_used=0,
            query=query,
        )

    # 2. Re-rank
    chunks = rerank_chunks(query, chunks)

    # 3. Generate answer
    answer = call_llm(query, chunks, history)

    # 4. Parse sections
    sections = parse_sections(answer)

    # 5. Follow-ups
    followups = generate_followups(query, answer)

    # 6. Citations (numbered to match [^n] markers)
    citations = extract_citations(chunks)

    return RAGResponse(
        answer=answer,
        sections=sections,
        citations=citations,
        chunks_used=len(chunks),
        followups=followups,
        query=query,
    )