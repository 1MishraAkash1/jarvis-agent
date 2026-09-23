"""
Phase 2 — searching your indexed notes at chat time.

agent.py calls `retrieve_relevant_chunks()` before sending your message to
the model. It embeds your question the same way index_notes.py embedded
your notes, then asks Chroma for the closest-matching chunks.

We only hand the model chunks that are ACTUALLY close matches (cosine
distance below DISTANCE_THRESHOLD) — for a casual message like "hello"
nothing in your notes will be close enough, so no notes get pulled in.
That keeps unrelated chat fast and avoids confusing the model with
irrelevant context it wasn't asked about.
"""

import glob
import os

import chromadb
import ollama

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma")
NOTES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "notes")
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "notes"

TOP_K = 3                 # how many chunks to consider per question
DISTANCE_THRESHOLD = 0.6  # cosine distance; lower = more similar, 0 = identical
                           # 0.5 was too strict (dropped a genuine match at 0.532);
                           # 0.7 was too loose (matched unrelated "tell me about
                           # Trump" at 0.62-0.67). 0.6 sits between the two.


def retrieve_relevant_chunks(query):
    """Returns the text of up to TOP_K note chunks relevant to `query`,
    or an empty list if nothing in the notes is a close enough match."""
    try:
        client = chromadb.PersistentClient(path=DB_DIR)
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        # No index built yet (or notes/ was empty when it was built).
        return []

    query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)

    chunks = []
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for doc, distance in zip(documents, distances):
        if distance <= DISTANCE_THRESHOLD:
            chunks.append(doc)
    return chunks


def load_all_notes_text():
    """Returns every note's raw content, for deterministic full-recall
    answers (broad "what do you know about me" questions). Deliberately
    bypasses chunking and similarity search — for a broad recall we want
    ALL notes verbatim, not a similarity-ranked subset, and no model
    synthesis step that could drop or garble content."""
    paths = glob.glob(os.path.join(NOTES_DIR, "*.txt")) + glob.glob(os.path.join(NOTES_DIR, "*.md"))
    if not paths:
        return ""
    lines = []
    for path in sorted(paths):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            lines.append(f"- ({os.path.basename(path)}) {content}")
    if not lines:
        return ""
    return "From your notes:\n" + "\n".join(lines)