"""
Hybrid PDF Pipeline  +  Ollama LLM Integration
================================================
1. Extract normal text using pdfplumber (+ PyMuPDF fallback)
2. Extract tables using camelot
3. Convert tables to markdown
4. Save tables as separate chunks
5. Store all chunks in a vector DB (ChromaDB)
6. Highlight table regions in the PDF (yellow highlight overlay)
7. [NEW] Ollama LLM integration:
      • Summarize every table with the LLM
      • RAG: answer questions about the PDF using retrieved chunks
      • Interactive Q&A CLI

Requirements:
    pip install pdfplumber pymupdf camelot-py[cv] chromadb pandas ollama

Install & run Ollama:
    https://ollama.com/download
    ollama pull llama3          # or mistral, phi3, gemma2, etc.
    ollama serve                # starts API on http://localhost:11434
"""

import os
import json
import uuid
import argparse
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber
import camelot
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import ollama  # official Ollama Python client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "llama3"       # change to any model you have pulled


# ---------------------------------------------------------------------------
# 1. Text Extraction
# ---------------------------------------------------------------------------

def extract_text_chunks(pdf_path: str) -> list[dict]:
    """
    Extract plain text page-by-page using pdfplumber.
    Falls back to PyMuPDF if pdfplumber returns nothing for a page.
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
                chunks.append({
                    "id":      str(uuid.uuid4()),
                    "type":    "text",
                    "page":    page_num,
                    "content": text.strip(),
                    "summary": None,
                })

    doc_fitz.close()
    return chunks


# ---------------------------------------------------------------------------
# 2 & 3. Table Extraction + Markdown Conversion
# ---------------------------------------------------------------------------

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a pandas DataFrame to a GitHub-flavoured markdown table."""
    df = df.fillna("").astype(str)
    headers   = list(df.iloc[0]) if not df.empty else list(df.columns)
    rows      = df.iloc[1:].values.tolist() if not df.empty else []
    header_row = "| " + " | ".join(headers) + " |"
    separator  = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows  = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator] + body_rows)


def extract_table_chunks(pdf_path: str, flavor: str = "lattice") -> list[dict]:
    """Extract tables using camelot and return as markdown chunks."""
    chunks = []
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor=flavor)
    except Exception as e:
        print(f"[camelot] {flavor} mode failed: {e}. Trying stream ...")
        try:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        except Exception as e2:
            print(f"[camelot] stream also failed: {e2}")
            return chunks

    for table in tables:
        df = table.df
        if df.empty:
            continue
        chunks.append({
            "id":        str(uuid.uuid4()),
            "type":      "table",
            "page":      table.page,
            "content":   dataframe_to_markdown(df),
            "bbox":      table._bbox,
            "dataframe": df,
            "summary":   None,   # filled by LLM later
        })
    return chunks


# ---------------------------------------------------------------------------
# 4. Save Chunks to JSON
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], output_path: str) -> None:
    serialisable = []
    for c in chunks:
        item = {k: v for k, v in c.items() if k != "dataframe"}
        if "bbox" in item and item["bbox"] is not None:
            item["bbox"] = list(item["bbox"])
        serialisable.append(item)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)
    print(f"[save_chunks] {len(serialisable)} chunks -> {output_path}")


# ---------------------------------------------------------------------------
# 5. Store in Vector DB (ChromaDB)
# ---------------------------------------------------------------------------

def store_in_vector_db(
    chunks: list[dict],
    collection_name: str = "pdf_chunks",
    persist_dir: str = "./chroma_db",
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )

    ids       = [c["id"]      for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [
        {
            "type":    c["type"],
            "page":    str(c["page"]),
            "bbox":    json.dumps(list(c["bbox"])) if c.get("bbox") else "",
            "summary": c.get("summary") or "",
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
    color: tuple = (1, 0.9, 0),
    opacity: float = 0.35,
) -> None:
    """Draw semi-transparent yellow rectangles over detected table regions."""
    doc = fitz.open(pdf_path)

    for chunk in table_chunks:
        if not chunk.get("bbox"):
            continue

        page_index = chunk["page"] - 1
        page       = doc[page_index]
        page_h     = page.rect.height
        x1, y1_cam, x2, y2_cam = chunk["bbox"]

        # camelot origin: bottom-left -> PyMuPDF origin: top-left
        fitz_y1 = page_h - y2_cam
        fitz_y2 = page_h - y1_cam

        rect  = fitz.Rect(x1, fitz_y1, x2, fitz_y2)
        annot = page.add_rect_annot(rect)
        annot.set_colors(stroke=color, fill=color)
        annot.set_opacity(opacity)
        annot.update()

        label_rect = fitz.Rect(x1, fitz_y1 - 14, x1 + 140, fitz_y1)
        page.insert_textbox(
            label_rect, "Table Detected",
            fontsize=8, color=(0.6, 0.3, 0),
            align=fitz.TEXT_ALIGN_LEFT,
        )

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"[highlight] Highlighted PDF -> {output_path}")


# ---------------------------------------------------------------------------
# 7. Ollama LLM Integration
# ---------------------------------------------------------------------------

class OllamaLLM:
    """Thin wrapper around the ollama Python client."""

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        self._check_connection()

    def _check_connection(self):
        try:
            models    = ollama.list()
            available = [m["name"] for m in models.get("models", [])]
            if not available:
                print("[ollama]  No models found. Run: ollama pull llama3")
                return
            matched = any(self.model in m for m in available)
            if not matched:
                print(f"[ollama]  Model '{self.model}' not found locally.")
                print(f"[ollama]  Available: {available}")
                print(f"[ollama]  Run: ollama pull {self.model}")
            else:
                print(f"[ollama]  Connected. Using model: {self.model}")
        except Exception as e:
            print(f"[ollama]  Cannot reach Ollama server: {e}")
            print("[ollama]  Make sure Ollama is running: ollama serve")

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Send a chat request and return the assistant reply."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                options={"temperature": temperature},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"[LLM Error] {e}"

    def summarize_table(self, markdown_table: str, page: int) -> str:
        """Ask the LLM to summarise a markdown table in 2-3 sentences."""
        system = (
            "You are a precise document analyst. "
            "Given a markdown table extracted from a PDF, write a concise "
            "2-3 sentence summary capturing the key data, patterns, or insights. "
            "Do NOT reproduce the raw table data."
        )
        user = (
            f"This table was found on page {page} of the document.\n\n"
            f"{markdown_table}\n\nPlease summarise this table."
        )
        return self.chat(system, user)

    def answer_with_context(self, question: str, context_chunks: list[dict]) -> str:
        """RAG-style answer: inject retrieved chunks as context."""
        parts = []
        for i, chunk in enumerate(context_chunks, 1):
            ctype = chunk["metadata"]["type"]
            page  = chunk["metadata"]["page"]
            parts.append(f"[Chunk {i} | type={ctype} | page={page}]\n{chunk['content']}")
        context_str = "\n\n---\n\n".join(parts)

        system = (
            "You are a helpful assistant answering questions about a PDF document. "
            "Use only the provided context to answer. "
            "If the context lacks enough information, say so. "
            "When referencing a table, mention its page number."
        )
        user = (
            f"Context from the document:\n\n{context_str}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        return self.chat(system, user, temperature=0.1)

    def is_relevant(self, question: str, chunk_content: str) -> bool:
        """Quick yes/no filter: is this chunk relevant to the question?"""
        system = "You are a relevance classifier. Reply only YES or NO."
        user   = (
            f"Question: {question}\n\n"
            f"Document excerpt:\n{chunk_content[:600]}\n\n"
            "Is this excerpt relevant to answering the question? (YES/NO)"
        )
        return self.chat(system, user, temperature=0.0).strip().upper().startswith("Y")


# ---------------------------------------------------------------------------
# RAG Query Engine
# ---------------------------------------------------------------------------

class RAGQueryEngine:
    """Combines ChromaDB retrieval with Ollama LLM to answer questions."""

    def __init__(
        self,
        collection: chromadb.Collection,
        llm: OllamaLLM,
        top_k: int = 5,
        rerank: bool = True,
    ):
        self.collection = collection
        self.llm        = llm
        self.top_k      = top_k
        self.rerank     = rerank

    def retrieve(self, query: str) -> list[dict]:
        results = self.collection.query(query_texts=[query], n_results=self.top_k)
        return [
            {
                "content":  doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i, doc in enumerate(results["documents"][0])
        ]

    def ask(self, question: str) -> dict:
        """
        Full RAG pipeline:
          1. Retrieve top-k chunks from ChromaDB
          2. Optional LLM re-rank to filter irrelevant chunks
          3. Generate grounded answer with context
        """
        print(f"\n[RAG] Retrieving chunks for: '{question}'")
        chunks = self.retrieve(question)

        if self.rerank:
            print(f"[RAG] Re-ranking {len(chunks)} chunks with LLM ...")
            chunks = [c for c in chunks if self.llm.is_relevant(question, c["content"])]
            print(f"[RAG] {len(chunks)} chunks kept after re-ranking")

        if not chunks:
            return {
                "question": question,
                "answer":   "No relevant content found in the document for this question.",
                "sources":  [],
            }

        answer = self.llm.answer_with_context(question, chunks)
        return {
            "question": question,
            "answer":   answer,
            "sources":  [
                {
                    "page":     c["metadata"]["page"],
                    "type":     c["metadata"]["type"],
                    "distance": round(c["distance"], 4),
                    "excerpt":  c["content"][:200],
                }
                for c in chunks
            ],
        }


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    pdf_path: str,
    output_dir: str = "./output",
    camelot_flavor: str = "lattice",
    db_dir: str = "./chroma_db",
    ollama_model: str = OLLAMA_MODEL,
    summarize_tables: bool = True,
) -> tuple[dict[str, Any], RAGQueryEngine]:
    """
    Run the full hybrid pipeline with Ollama LLM.

    Returns:
        (summary_dict, rag_engine)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base_name = Path(pdf_path).stem

    banner = f"  Hybrid PDF + Ollama Pipeline  |  {Path(pdf_path).name}  "
    print(f"\n{'='*len(banner)}\n{banner}\n{'='*len(banner)}\n")

    # Init LLM
    print("Initialising Ollama LLM ...")
    llm = OllamaLLM(model=ollama_model)

    # Step 1: Text
    print("\nStep 1  Extracting text ...")
    text_chunks = extract_text_chunks(pdf_path)
    print(f"        -> {len(text_chunks)} text chunks")

    # Steps 2+3: Tables
    print("\nStep 2/3  Extracting tables -> markdown ...")
    table_chunks = extract_table_chunks(pdf_path, flavor=camelot_flavor)
    print(f"          -> {len(table_chunks)} table chunks")

    # Step 7a: Summarise tables
    if summarize_tables and table_chunks:
        print(f"\nStep 7a  Summarising {len(table_chunks)} table(s) with LLM ...")
        for i, chunk in enumerate(table_chunks):
            print(f"  [{i+1}/{len(table_chunks)}] Page {chunk['page']} ...")
            chunk["summary"] = llm.summarize_table(chunk["content"], chunk["page"])
            print(f"    -> {chunk['summary'][:100]} ...")

    all_chunks = text_chunks + table_chunks

    # Step 4: Save
    print("\nStep 4  Saving chunks ...")
    chunks_path = os.path.join(output_dir, f"{base_name}_chunks.json")
    save_chunks(all_chunks, chunks_path)

    # Step 5: Vector DB
    print("\nStep 5  Storing in ChromaDB ...")
    collection = store_in_vector_db(
        all_chunks,
        collection_name=base_name.replace(" ", "_"),
        persist_dir=db_dir,
    )

    # Step 6: Highlight PDF
    print("\nStep 6  Highlighting tables in PDF ...")
    highlighted_path = os.path.join(output_dir, f"{base_name}_highlighted.pdf")
    highlight_tables_in_pdf(pdf_path, table_chunks, highlighted_path)

    # Step 7b: RAG engine
    print("\nStep 7b  Building RAG query engine ...")
    rag_engine = RAGQueryEngine(collection=collection, llm=llm, top_k=5)
    print("         -> Ready")

    summary = {
        "source_pdf":      pdf_path,
        "text_chunks":     len(text_chunks),
        "table_chunks":    len(table_chunks),
        "total_chunks":    len(all_chunks),
        "chunks_json":     chunks_path,
        "highlighted_pdf": highlighted_path,
        "vector_db_dir":   db_dir,
        "collection_name": collection.name,
        "llm_model":       ollama_model,
    }

    print(f"\n{'='*50}\n  Pipeline complete!\n{'='*50}")
    for k, v in summary.items():
        print(f"  {k:<22} {v}")
    print(f"{'='*50}\n")

    return summary, rag_engine


# ---------------------------------------------------------------------------
# Interactive Q&A CLI
# ---------------------------------------------------------------------------

def interactive_qa(rag_engine: RAGQueryEngine) -> None:
    print("\n" + "="*60)
    print("  Interactive Q&A  (type 'exit' to quit)")
    print("="*60)

    while True:
        try:
            question = input("\n? Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question or question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        result = rag_engine.ask(question)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources ({len(result['sources'])} chunks):")
        for s in result["sources"]:
            print(
                f"  Page {s['page']} [{s['type']}]  dist={s['distance']}  "
                f"-> {s['excerpt'][:80].replace(chr(10),' ')} ..."
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid PDF Pipeline with Ollama LLM"
    )
    parser.add_argument("pdf",            help="Path to the PDF file")
    parser.add_argument("--flavor",       default="lattice",
                        choices=["lattice", "stream"],
                        help="camelot table extraction mode (default: lattice)")
    parser.add_argument("--model",        default=OLLAMA_MODEL,
                        help=f"Ollama model (default: {OLLAMA_MODEL})")
    parser.add_argument("--output-dir",   default="./output")
    parser.add_argument("--db-dir",       default="./chroma_db")
    parser.add_argument("--no-summarize", action="store_true",
                        help="Skip LLM table summarisation")
    parser.add_argument("--question",     default=None,
                        help="Ask a single question (skips interactive mode)")
    args = parser.parse_args()

    summary, rag = run_pipeline(
        pdf_path         = args.pdf,
        output_dir       = args.output_dir,
        camelot_flavor   = args.flavor,
        db_dir           = args.db_dir,
        ollama_model     = args.model,
        summarize_tables = not args.no_summarize,
    )

    if args.question:
        result = rag.ask(args.question)
        print(f"\n? {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        for s in result["sources"]:
            print(f"  Page {s['page']} [{s['type']}]")
    else:
        interactive_qa(rag)