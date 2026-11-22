"""
pipeline.py
-----------
Chunking, vectorisation, and media extraction pipeline for bank annual report PDFs
and XLS financial supplements.

Extracts:
  - Text chunks  → FAISS vector index (PDF + XLS)
  - Tables       → stored as JSON per page in page_assets/
  - Table index  → searchable table catalogue in page_assets/table_index.json
  - Images       → stored as PNG per page in page_assets/

Improvements over base version:
  - Header/footer noise cleaning (strips page numbers, repeated bank names)
  - Section inference (detects Basel III table codes e.g. CR1, MR1, OV1)
  - XLS ingestion (serialises spreadsheet rows as natural language for embedding)

Usage:
    python -m app.pipeline --pdfs natwest.pdf lloyds.pdf
    python -m app.pipeline --pdfs natwest.pdf lloyds.pdf --xls natwest_supplement.xlsx lloyds_supplement.xlsx
"""
import argparse
import json
import os
import re
from pathlib import Path

import pdfplumber


import pandas as pd
import openpyxl
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHUNK_SIZE    = 2048   # ~512 tokens — sharper embeddings than 800 chars
CHUNK_OVERLAP = 256    # ~64 tokens of context overlap
INDEX_DIR     = os.path.join(os.path.dirname(__file__), '..', 'faiss_index')
ASSETS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'page_assets')

# Separator hierarchy tuned for regulatory document structure
SEPARATORS = [
    '\n#{1,3} ',    # Markdown headings
    '\n\n',         # Paragraph breaks
    '\n',           # Line breaks
    r'(?<=\.) ',    # Sentence boundaries
    ' ',            # Word boundaries (last resort)
]

# ---------------------------------------------------------------------------
# Noise cleaning (from ingest_pdf.py)
# ---------------------------------------------------------------------------

# Regex matching lines that are pure header/footer noise on regulatory PDFs:
# bare page numbers, repeated bank names, boilerplate labels
HEADER_FOOTER_RE = re.compile(
    r'^(page\s*\d+'
    r'|\d+'
    r'|confidential'
    r'|restricted'
    r'|lloyds'
    r'|natwest'
    r'|hsbc'
    r'|barclays'
    r'|standard chartered'
    r'|santander'
    r')$',
    re.IGNORECASE
)

# Basel III disclosure table code pattern e.g. CR1, MR1, OV1, LCR1
BASEL_TABLE_RE    = re.compile(r'^[A-Z]{2,4}\d+[A-Z]?$')
MIN_HEADING_LENGTH = 5

# Patterns indicating a chunk contains tabular data
TABLE_INDICATOR_RE = re.compile(
    r'\|'           # pipe characters
    r'|\t'          # tab-separated
    r'|=\s*[\d,\.]+'  # key=value pairs
    r'|\d+\s*%'     # percentage values
)

# Minimum chunk character length — discard noise fragments below this
MIN_CHUNK_CHARS = 50


def _clean_text(text: str) -> str:
    """
    Remove header/footer noise from a page's extracted text.
    Strips page numbers, repeated bank names, and boilerplate labels.
    """
    lines = [
        line for line in text.splitlines()
        if line.strip() and not HEADER_FOOTER_RE.match(line.strip())
    ]
    return '\n'.join(lines).strip()


def _infer_section(text: str) -> str:
    """
    Infer the regulatory section label for a page.
    Priority: Basel III table code > all-caps heading > 'body'
    """
    for line in text.splitlines():
        stripped = line.strip()
        if BASEL_TABLE_RE.match(stripped):
            return stripped
        if (stripped
                and stripped.isupper()
                and len(stripped) > MIN_HEADING_LENGTH
                and not stripped.isdigit()):
            return stripped
    return 'body'

def _extract_tables_pdfplumber(page) -> list:
    """
    Extract tables from a pdfplumber `page` object using pdfplumber's
    `extract_tables()` API. Returns a list of cleaned tables (list of rows).
    """
    tables = []
    try:
        raw_tables = page.extract_tables()
        for tbl in raw_tables or []:
            clean = [
                [ (cell if cell is not None else "").strip() for cell in row ]
                for row in tbl
            ]
            if len(clean) > 1:
                tables.append(clean)
    except Exception:
        pass

    return tables

def load_pdf(pdf_path: str) -> tuple[list[Document], dict]:
    """
    Load a PDF using pdfplumber with noise cleaning and section inference.
    Returns:
      - pages:  list of LangChain Documents (one per page)
      - assets: dict mapping (source_doc, page_num) → {"tables": [...], "image_path": str|None}
    """
    print(f"  Loading PDF: {pdf_path}")
    doc_name = Path(pdf_path).stem.replace("_", " ").replace("-", " ").title()
    pages = []
    assets = {}
    os.makedirs(ASSETS_DIR, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            page_assets = {"tables": [], "image_path": None}

            # --- Text (with noise cleaning) ---
            raw = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            text = _clean_text(raw)

            if not text:
                # Skip blank/noise-only pages (cover pages, dividers)
                continue

            section = _infer_section(text)

            pages.append(Document(
                page_content=text,
                metadata={
                    "source_doc": doc_name,
                    "page": page_num,
                    "section": section,
                    "doc_type": "pdf",
                },
            ))

            # --- Tables ---
            try:
                page_assets["tables"] = _extract_tables_pdfplumber(page)
            except Exception as e:
                print(f"    Warning: table extraction failed on p.{page_num}: {e}")

            # --- Page image ---
            try:
                img = page.to_image(resolution=120)
                safe_name = doc_name.replace(" ", "_")
                img_filename = f"{safe_name}_p{page_num}.png"
                img_path = os.path.join(ASSETS_DIR, img_filename)
                img.save(img_path)
                page_assets["image_path"] = img_path
            except Exception as e:
                print(f"    Warning: image extraction failed on p.{page_num}: {e}")

            assets[(doc_name, page_num)] = page_assets

            if page_assets["tables"]:
                _save_table_index(doc_name, page_num, page_assets["tables"])

            if page_num % 20 == 0 or page_num == total:
                print(f"    Processed {page_num}/{total} pages...")

    _save_assets_index(assets)
    print(f"    → {len(pages)} pages loaded from {doc_name}")
    return pages, assets


# ---------------------------------------------------------------------------
# Step 1b: Load XLS — serialise rows as natural language Documents
# ---------------------------------------------------------------------------

# Basel III tab names to process per bank. Add/adjust as needed.
XLS_TAB_MAP: dict[str, list[str]] = {
    'lloyds':  ['OV1', 'CR1', 'CR2', 'CR4', 'MR1', 'LR1', 'LCR'],
    'natwest': ['OV1', 'CR1', 'CR2', 'MR1', 'LR1'],
}


def _normalise_tab_name(name: str) -> str:
    """Normalise a worksheet tab name for matching against XLS_TAB_MAP."""
    name = name.strip().upper()
    if name.startswith('TABLE '):
        name = name[6:].strip()
    return name.split(' ')[0].split('-')[0].strip()


def _row_to_text(bank: str, year: int, tab: str, headers: list, row: list) -> str:
    """
    Serialise a single XLS row into a natural language string for embedding.

    Example output:
      "Lloyds 2025 CR1: Residential mortgages: EAD = 142300, RWA = 118200, Ratio = 14.2%"
    """
    pairs = []
    for header, value in zip(headers, row):
        if header and value is not None and str(value).strip() not in ('', 'nan', 'None'):
            pairs.append(f"{header} = {value}")

    if len(pairs) < 1:
        return ""

    label = str(row[0]).strip() if row else ""
    data  = ', '.join(pairs[1:]) if len(pairs) > 1 else pairs[0]
    return f"{bank} {year} {tab}: {label}: {data}"


def load_xls(xls_path: str, bank_hint: str = None) -> list[Document]:
    """
    Load an XLS/XLSX financial supplement and convert each data row into
    a LangChain Document for embedding.

    Args:
        xls_path:  Path to the .xlsx file
        bank_hint: Optional bank name override (inferred from filename if not given)

    Returns:
        List of LangChain Documents, one per meaningful data row
    """
    print(f"  Loading XLS: {xls_path}")
    path      = Path(xls_path)
    doc_name  = path.stem.replace("_", " ").replace("-", " ").title()
    bank      = (bank_hint or path.stem.split("_")[0]).lower()
    year_match = re.search(r'(20\d{2})', path.stem)
    year      = int(year_match.group(1)) if year_match else 2025
    documents = []

    allowed_tabs = [t.upper() for t in XLS_TAB_MAP.get(bank, [])]

    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except Exception as e:
        print(f"  ERROR: Cannot open {xls_path}: {e}")
        return []

    for sheet_name in wb.sheetnames:
        normalised = _normalise_tab_name(sheet_name)

        # Skip tabs not in the allowed list (if a list is defined)
        if allowed_tabs and normalised not in allowed_tabs:
            continue

        print(f"    Processing tab: {sheet_name}")
        try:
            df = pd.read_excel(str(path), sheet_name=sheet_name, header=None)
            if df.empty:
                continue

            # Find the header row — first row with at least 2 non-null values
            header_row_idx = 0
            for idx, row in df.iterrows():
                if row.notna().sum() >= 2:
                    header_row_idx = idx
                    break

            headers = [
                str(v).strip() if pd.notna(v) else ""
                for v in df.iloc[header_row_idx]
            ]

            row_count = 0
            for _, row in df.iloc[header_row_idx + 1:].iterrows():
                row_vals = list(row)
                text     = _row_to_text(doc_name, year, normalised, headers, row_vals)
                if not text or len(text) < 20:
                    continue

                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source_doc": doc_name,
                        "page":       0,          # XLS has no page concept
                        "section":    normalised,
                        "doc_type":   "xls",
                        "table_id":   f"{bank}_{year}_{normalised}",
                    },
                ))
                row_count += 1

            print(f"      → {row_count} rows extracted")

        except Exception as e:
            print(f"    Warning: could not parse tab '{sheet_name}': {e}")

    wb.close()
    print(f"    → {len(documents)} total XLS documents from {doc_name}")
    return documents


# ---------------------------------------------------------------------------
# Table index (for synthesis)
# ---------------------------------------------------------------------------

def _table_index_path() -> str:
    return os.path.join(ASSETS_DIR, "table_index.json")


def _save_table_index(doc_name: str, page_num: int, tables: list):
    index_path = _table_index_path()
    existing   = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            existing = json.load(f)

    existing = [e for e in existing if not (e["doc"] == doc_name and e["page"] == page_num)]

    for i, table in enumerate(tables):
        if not table or len(table) < 2:
            continue
        headers = table[0]
        rows    = table[1:]
        existing.append({
            "doc":       doc_name,
            "page":      page_num,
            "table_idx": i,
            "headers":   headers,
            "rows":      rows,
            "preview":   f"{doc_name} p.{page_num} | Columns: {', '.join(str(h) for h in headers if h)}",
        })

    with open(index_path, "w") as f:
        json.dump(existing, f)


def load_table_index() -> list:
    path = _table_index_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


# ---------------------------------------------------------------------------
# Assets index helpers
# ---------------------------------------------------------------------------

def _assets_index_path() -> str:
    return os.path.join(ASSETS_DIR, "assets_index.json")


def _save_assets_index(assets: dict):
    os.makedirs(ASSETS_DIR, exist_ok=True)
    serialisable = {
        f"{doc}|||{page}": val
        for (doc, page), val in assets.items()
    }
    existing = _load_assets_index()
    existing.update(serialisable)
    with open(_assets_index_path(), "w") as f:
        json.dump(existing, f)


def _load_assets_index() -> dict:
    path = _assets_index_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_assets_for_pages(citations: list[dict]) -> list[dict]:
    index   = _load_assets_index()
    results = []
    seen    = set()
    for c in citations:
        key = f"{c['doc']}|||{c['page']}"
        if key in index and key not in seen:
            seen.add(key)
            entry         = dict(index[key])
            entry["doc"]  = c["doc"]
            entry["page"] = c["page"]
            results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Step 2: Chunk
# ---------------------------------------------------------------------------

def chunk_documents(pages: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks.
    XLS documents are typically short enough to pass through unchanged.
    PDF pages are split using a regulatory-tuned separator hierarchy.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size         = CHUNK_SIZE,
        chunk_overlap      = CHUNK_OVERLAP,
        separators         = SEPARATORS,
        length_function    = len,
        is_separator_regex = True,
    )

    # Filter out very short chunks (noise/blank pages)
    raw_chunks = splitter.split_documents(pages)
    chunks     = [c for c in raw_chunks if len(c.page_content) >= MIN_CHUNK_CHARS]

    print(f"    → {len(chunks)} chunks created ({len(raw_chunks) - len(chunks)} noise chunks discarded)")
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Embed + store
# ---------------------------------------------------------------------------

def _get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_vector_store(chunks: list[Document], index_dir: str):
    print(f"Embedding {len(chunks)} chunks and building FAISS index...")
    embeddings   = _get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(index_dir)
    print(f"  → Done. Index saved to {index_dir}/")
    return vector_store


def load_vector_store(index_dir: str = INDEX_DIR):
    embeddings   = _get_embeddings()
    vector_store = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"Loaded FAISS index from {index_dir}/")
    return vector_store


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    pdf_paths:  list[str],
    xls_paths:  list[str] = None,
    index_dir:  str       = INDEX_DIR,
):
    print("=" * 50)
    print("Bank Report Chunking & Vectorisation Pipeline")
    print("=" * 50)

    all_chunks = []

    # --- PDFs ---
    for pdf_path in (pdf_paths or []):
        if not os.path.exists(pdf_path):
            print(f"WARNING: File not found, skipping: {pdf_path}")
            continue
        print(f"\nProcessing PDF: {pdf_path}")
        pages, _ = load_pdf(pdf_path)
        chunks   = chunk_documents(pages)
        all_chunks.extend(chunks)

    # --- XLS ---
    for xls_path in (xls_paths or []):
        if not os.path.exists(xls_path):
            print(f"WARNING: File not found, skipping: {xls_path}")
            continue
        print(f"\nProcessing XLS: {xls_path}")
        pages  = load_xls(xls_path)
        chunks = chunk_documents(pages)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("\nNo chunks produced — check your file paths.")
        return None

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")
    vector_store = build_vector_store(all_chunks, index_dir)
    print("\nPipeline complete!")
    return vector_store


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk and vectorise bank annual report PDFs and XLS files")
    parser.add_argument("--pdfs",      nargs="*", default=[],    help="Paths to PDF files")
    parser.add_argument("--xls",       nargs="*", default=[],    help="Paths to XLS/XLSX files")
    parser.add_argument("--index-dir", default=INDEX_DIR,        help="Directory to save FAISS index")
    args = parser.parse_args()
    run_pipeline(args.pdfs, args.xls, args.index_dir)