"""
Phase 2 — building a searchable index of your notes.

Run this file (`python3 src/rag/index_notes.py`) any time you add, edit, or
remove files in notes/. It reads every .txt and .md file there, splits each
one into overlapping chunks, turns each chunk into an embedding vector
(using Ollama's nomic-embed-text model), and stores the result in a local
Chroma database on disk (data/chroma/). agent.py reads from that database
at chat time — it never re-reads your raw note files directly, so if you
edit a note, rerun this script before asking Jarvis about the change.

Why chunk instead of embedding a whole file at once?
  - Retrieval returns whole chunks, not files. Small, focused chunks mean a
    question about one paragraph of a long file returns just that
    paragraph, not the entire document diluted with unrelated content.
  - A 5,000-word file crammed into a single embedding loses precision
    compared to several smaller chunks each embedded on their own.

Why overlap chunks?
  - If a sentence that answers a question sits right at a chunk boundary,
    a hard cut could split it awkwardly across two chunks, weakening both.
    A small overlap (some text repeated at the start of the next chunk)
    makes that much less likely.
"""

import glob
import os

import chromadb
import ollama

NOTES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "notes")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma")
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "notes"

CHUNK_SIZE = 800     # characters per chunk (roughly 150-200 words)
CHUNK_OVERLAP = 100  # characters repeated between consecutive chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into overlapping fixed-size chunks.

    Simple character-based sliding window — good enough to start with.
    A more advanced version would split on sentence/paragraph boundaries
    instead of cutting mid-word; worth revisiting if retrieval quality
    disappoints in practice.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # step forward, but re-include the overlap
    return chunks


def load_notes():
    """Returns a list of (filename, full_text) for every .txt/.md file in notes/."""
    paths = glob.glob(os.path.join(NOTES_DIR, "*.txt")) + glob.glob(os.path.join(NOTES_DIR, "*.md"))
    notes = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            notes.append((os.path.basename(path), f.read()))
    return notes


def main():
    notes = load_notes()
    if not notes:
        print(f"No .txt or .md files found in {NOTES_DIR} — add some notes and rerun.")
        return

    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)

    # Delete and rebuild the whole collection each run — the simplest
    # correct behavior for Phase 2. A production version would only
    # re-embed files that actually changed; worth adding once your notes
    # collection is large enough that a full rebuild feels slow.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    doc_ids, doc_texts, doc_metadatas, doc_embeddings = [], [], [], []

    for filename, text in notes:
        chunks = chunk_text(text)
        print(f"{filename}: {len(chunks)} chunk(s)")
        if not chunks:
            print(f"  (skipping {filename} — file is empty, nothing to index)")
            continue
        for i, chunk in enumerate(chunks):
            embedding = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)["embedding"]
            doc_ids.append(f"{filename}::{i}")
            doc_texts.append(chunk)
            doc_metadatas.append({"source": filename, "chunk_index": i})
            doc_embeddings.append(embedding)

    if not doc_ids:
        print("\nNothing to index — every note file was empty. Add some text and rerun.")
        return

    collection.add(
        ids=doc_ids,
        documents=doc_texts,
        metadatas=doc_metadatas,
        embeddings=doc_embeddings,
    )
    print(f"\nIndexed {len(doc_ids)} chunk(s) from {len(notes)} file(s) into {DB_DIR}")


if __name__ == "__main__":
    main()