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

import os

import chromadb
import ollama

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma")
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "notes"

TOP_K = 3                 # how many chunks to consider per question
DISTANCE_THRESHOLD = 0.7  # cosine distance; lower = more similar, 0 = identical


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