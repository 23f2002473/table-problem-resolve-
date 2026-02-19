"""
Hybrid PDF Pipeline
====================
1. Extract normal text using pdfplumber (+ PyMuPDF fallback)
2. Extract tables using camelot
3. Convert tables to markdown
4. Save tables as separate chunks
5. Store all chunks in a vector DB (ChromaDB)
6. Highlight table regions in the PDF (yellow highlight overlay)

Requirements:
    pip install pdfplumber pymupdf camelot-py[cv] chromadb pandas

Note: camelot requires ghostscript for lattice mode.
      Install via: brew install ghostscript  /  apt install ghostscript
"""

import os
import json
import uuid
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber
import camelot
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions


# ---------------------------------------------------------------------------
# 1. Text Extraction
# ---------------------------------------------------------------------------

def extract_text_chunks(pdf_path: str) -> list[dict]:
    """
    Extract plain text page-by-page using pdfplumber.
    Falls back to PyMuPDF if pdfplumber returns nothing for a page.

    Returns a list of chunk dicts:
        {id, type, page, content}
    """
    chunks = []
    doc_fitz = fitz.open(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()

            # Fallback to PyMuPDF
            if not text or not text.strip():
                text = doc_fitz[page_num - 1].get_text("text")

            if text and text.strip():
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "type": "text",
                        "page": page_num,
                        "content": text.strip(),
                    }
                )

    doc_fitz.close()
    return chunks


# ---------------------------------------------------------------------------
# 2 & 3. Table Extraction + Markdown Conversion
# ---------------------------------------------------------------------------

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a pandas DataFrame to a GitHub-flavoured markdown table."""
    df = df.fillna("").astype(str)

    # Use first row as header if it looks like one, else use col indices
    headers = list(df.iloc[0]) if not df.empty else list(df.columns)
    rows = df.iloc[1:].values.tolist() if not df.empty else []

    header_row = "| " + " | ".join(headers) + " |"
    separator  = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows  = ["| " + " | ".join(row) + " |" for row in rows]

    return "\n".join([header_row, separator] + body_rows)


def extract_table_chunks(pdf_path: str, flavor: str = "lattice") -> list[dict]:
    """
    Extract tables using camelot (flavor: 'lattice' or 'stream').

    Returns a list of chunk dicts:
        {id, type, page, content (markdown), bbox, dataframe}
    """
    chunks = []
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor=flavor)
    except Exception as e:
        print(f"[camelot] {flavor} extraction failed: {e}. Trying stream mode…")
        try:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        except Exception as e2:
            print(f"[camelot] stream extraction also failed: {e2}")
            return chunks

    for table in tables:
        df = table.df
        if df.empty:
            continue

        markdown = dataframe_to_markdown(df)
        # camelot bbox: (x1, y1, x2, y2) in PDF coordinate space
        bbox = table._bbox  # (x1, y1, x2, y2)

        chunks.append(
            {
                "id": str(uuid.uuid4()),
                "type": "table",
                "page": table.page,
                "content": markdown,
                "bbox": bbox,          # used later for highlighting
                "dataframe": df,       # available for downstream use
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# 4. Save Chunks to JSON
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], output_path: str) -> None:
    """Persist chunks to a JSON file (dataframe stripped for serialisability)."""
    serialisable = []
    for c in chunks:
        item = {k: v for k, v in c.items() if k != "dataframe"}
        if "bbox" in item and item["bbox"] is not None:
            item["bbox"] = list(item["bbox"])
        serialisable.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)

    print(f"[save_chunks] Saved {len(serialisable)} chunks → {output_path}")


# ---------------------------------------------------------------------------
# 5. Store in Vector DB (ChromaDB)
# ---------------------------------------------------------------------------

def store_in_vector_db(
    chunks: list[dict],
    collection_name: str = "pdf_chunks",
    persist_dir: str = "./chroma_db",
) -> chromadb.Collection:
    """
    Upsert all chunks into a ChromaDB collection.
    Uses the default sentence-transformer embedding function.
    """
    client = chromadb.PersistentClient(path=persist_dir)

    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
    )

    ids       = [c["id"]      for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [
        {
            "type": c["type"],
            "page": str(c["page"]),
            "bbox": json.dumps(list(c["bbox"])) if c.get("bbox") else "",
        }
        for c in chunks
    ]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"[vector_db] Upserted {len(ids)} chunks into '{collection_name}'")
    return collection


# ---------------------------------------------------------------------------
# 6. Highlight Tables in PDF
# ---------------------------------------------------------------------------

def highlight_tables_in_pdf(
    pdf_path: str,
    table_chunks: list[dict],
    output_path: str,
    color: tuple = (1, 0.9, 0),          # yellow (R, G, B) 0-1 range
    opacity: float = 0.35,
) -> None:
    """
    Draw a semi-transparent yellow rectangle over every detected table.

    camelot returns bbox as (x1, y1, x2, y2) in PDF user-space where
    (0,0) is the BOTTOM-LEFT corner. PyMuPDF uses TOP-LEFT origin, so
    we convert: fitz_y = page_height - camelot_y.
    """
    doc = fitz.open(pdf_path)

    for chunk in table_chunks:
        if chunk.get("bbox") is None:
            continue

        page_index = chunk["page"] - 1          # 0-based
        page       = doc[page_index]
        page_h     = page.rect.height

        x1, y1_cam, x2, y2_cam = chunk["bbox"]

        # Convert camelot (bottom-left origin) → PyMuPDF (top-left origin)
        fitz_y1 = page_h - y2_cam
        fitz_y2 = page_h - y1_cam

        rect = fitz.Rect(x1, fitz_y1, x2, fitz_y2)

        # Draw filled rectangle annotation
        annot = page.add_rect_annot(rect)
        annot.set_colors(stroke=color, fill=color)
        annot.set_opacity(opacity)
        annot.update()

        # Optional: add a label above the highlight
        label_rect = fitz.Rect(x1, fitz_y1 - 14, x1 + 100, fitz_y1)
        page.insert_textbox(
            label_rect,
            "📊 Table",
            fontsize=8,
            color=(0.6, 0.3, 0),
            align=fitz.TEXT_ALIGN_LEFT,
        )

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"[highlight] Saved highlighted PDF → {output_path}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    pdf_path: str,
    output_dir: str = "./output",
    camelot_flavor: str = "lattice",
    db_dir: str = "./chroma_db",
) -> dict[str, Any]:
    """
    Run the full hybrid pipeline.

    Args:
        pdf_path:       Path to the source PDF.
        output_dir:     Directory for JSON chunks + highlighted PDF.
        camelot_flavor: 'lattice' (bordered tables) or 'stream' (borderless).
        db_dir:         ChromaDB persistence directory.

    Returns:
        Summary dict with paths and chunk counts.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base_name = Path(pdf_path).stem

    print(f"\n{'='*60}")
    print(f"  Hybrid PDF Pipeline  |  {Path(pdf_path).name}")
    print(f"{'='*60}\n")

    # --- Step 1: Text extraction ---
    print("Step 1 · Extracting text …")
    text_chunks = extract_text_chunks(pdf_path)
    print(f"         → {len(text_chunks)} text chunks extracted")

    # --- Step 2 & 3: Table extraction + markdown ---
    print("Step 2/3 · Extracting tables and converting to markdown …")
    table_chunks = extract_table_chunks(pdf_path, flavor=camelot_flavor)
    print(f"          → {len(table_chunks)} table chunks extracted")

    all_chunks = text_chunks + table_chunks

    # --- Step 4: Save chunks ---
    print("Step 4 · Saving chunks to JSON …")
    chunks_path = os.path.join(output_dir, f"{base_name}_chunks.json")
    save_chunks(all_chunks, chunks_path)

    # --- Step 5: Vector DB ---
    print("Step 5 · Storing chunks in ChromaDB …")
    collection = store_in_vector_db(
        all_chunks,
        collection_name=base_name.replace(" ", "_"),
        persist_dir=db_dir,
    )

    # --- Step 6: Highlight PDF ---
    print("Step 6 · Highlighting tables in PDF …")
    highlighted_path = os.path.join(output_dir, f"{base_name}_highlighted.pdf")
    highlight_tables_in_pdf(pdf_path, table_chunks, highlighted_path)

    summary = {
        "source_pdf":        pdf_path,
        "text_chunks":       len(text_chunks),
        "table_chunks":      len(table_chunks),
        "total_chunks":      len(all_chunks),
        "chunks_json":       chunks_path,
        "highlighted_pdf":   highlighted_path,
        "vector_db_dir":     db_dir,
        "collection_name":   collection.name,
    }

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    for k, v in summary.items():
        print(f"  {k:<20} {v}")
    print(f"{'='*60}\n")

    return summary


# ---------------------------------------------------------------------------
# Example query helper
# ---------------------------------------------------------------------------

def query_vector_db(
    query: str,
    collection_name: str,
    db_dir: str = "./chroma_db",
    n_results: int = 5,
) -> list[dict]:
    """Query the ChromaDB collection for the most relevant chunks."""
    client = chromadb.PersistentClient(path=db_dir)
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(name=collection_name, embedding_function=ef)

    results = collection.query(query_texts=[query], n_results=n_results)
    hits = []
    for i, doc in enumerate(results["documents"][0]):
        hits.append(
            {
                "rank":     i + 1,
                "content":  doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return hits


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hybrid_pdf_pipeline.py <path/to/document.pdf> [lattice|stream]")
        sys.exit(1)

    pdf_file   = sys.argv[1]
    flavor     = sys.argv[2] if len(sys.argv) > 2 else "lattice"
    result     = run_pipeline(pdf_file, camelot_flavor=flavor)

    # Demo: query for tables
    print("\nDemo query: 'table data summary'")
    hits = query_vector_db(
        "table data summary",
        collection_name=Path(pdf_file).stem.replace(" ", "_"),
    )
    for h in hits:
        print(f"  Rank {h['rank']} | type={h['metadata']['type']} | page={h['metadata']['page']}")
        print(f"    {h['content'][:120].replace(chr(10),' ')} …\n")