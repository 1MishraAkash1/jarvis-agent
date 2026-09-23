"""
Phase 3 — persistent memory (retrieval-based).

v1 of this file injected EVERY saved fact into the system prompt on every
message. That has two real problems, both confirmed by hand:
  1. Scaling: prompt size grows forever as you save more facts, and Phase 1
     proved prompt size is the main lever on response speed on this
     hardware. Hundreds of facts would slow down even "hello".
  2. Reliability: a small local model doesn't reliably scan a long, static
     "known facts" block when answering a broad question — it gravitates
     to whatever's most directly relevant to the current message. We saw
     this directly: "what do you know about me?" missed a fact that WAS in
     the prompt, while "what type of answers do I prefer?" found it fine.

Fix: treat facts like Phase 2 treats notes. Embed each fact when it's
saved, and at question time retrieve only the facts relevant to THAT
question — bounded size regardless of how many facts exist, and a shorter,
targeted context a small model can actually use.

One exception: a genuinely broad question ("what do you know about me?")
isn't asking about one topic — similarity search naturally favors facts
that are topically close to the wording of the question, not an exhaustive
dump. For those, we deliberately bypass retrieval and return everything.
"""

import json
import os
from datetime import datetime

import chromadb
import ollama

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "memory.json")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma")
COLLECTION_NAME = "memory_facts"
EMBED_MODEL = "nomic-embed-text"

TOP_K = 5                 # how many facts to consider per question
DISTANCE_THRESHOLD = 0.6  # cosine distance; lower = more similar, 0 = identical
                           # matched to the same value in rag/retrieve.py after
                           # 0.7 proved too loose there for unrelated queries

# Phrases that signal the user wants a COMPLETE recall, not one specific
# fact. For these we skip similarity search entirely and return every saved
# fact, because "what do you know about me" isn't topically close to any
# one fact in particular — it's asking for all of them.
BROAD_RECALL_PHRASES = [
    "what do you know about me",
    "tell me about myself",
    "what have you learned about me",
    "everything you know about me",
    "what do you remember about me",
    "what do you know about myself",
]


def _load_facts():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []  # corrupt/empty file — treat as no memory rather than crash


def _save_facts(facts):
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)


def _get_memory_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def remember_fact(fact: str) -> str:
    """Saves a fact about the user permanently, so Jarvis remembers it in future sessions too.

    Use this when the user explicitly asks you to remember something, or
    states a stable personal fact worth keeping long-term (their name, a
    preference, an ongoing project) — not for one-off details only
    relevant to the current message. IMPORTANT: the fact must come from
    what the user actually typed in their current message, never from
    background/retrieved notes shown for context.

    Args:
        fact: The fact to remember, as a short plain statement, e.g. "User's name is Akash".
    """
    facts = _load_facts()

    # Dedup: don't save a fact that's already stored, near-identical text.
    normalized_new = fact.strip().lower()
    for entry in facts:
        if entry["fact"].strip().lower() == normalized_new:
            return f"Already remembered: {fact}"

    facts.append({"fact": fact, "saved_at": datetime.now().isoformat(timespec="seconds")})
    _save_facts(facts)

    # Also index this fact for retrieval (best-effort — the JSON file above
    # is the source of truth; if embedding fails for any reason, the fact
    # is still saved and will still appear in a full/broad-recall dump).
    try:
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=fact)["embedding"]
        collection = _get_memory_collection()
        collection.add(ids=[f"fact-{len(facts) - 1}"], documents=[fact], embeddings=[embedding])
    except Exception:
        pass

    return f"Saved to memory: {fact}"


def is_broad_recall_query(query: str) -> bool:
    """True if the question is asking for an exhaustive recall ("what do you
    know about me") rather than one specific fact — these need special
    handling since similarity search alone would under-return for them."""
    q = query.lower()
    return any(phrase in q for phrase in BROAD_RECALL_PHRASES)


def load_all_facts_text() -> str:
    """Returns every saved fact as plain text — used for broad-recall
    questions where we deliberately bypass similarity-based retrieval.
    Empty string if nothing has been saved yet."""
    facts = _load_facts()
    if not facts:
        return ""
    lines = [f"- {entry['fact']}" for entry in facts]
    return "Known facts about the user:\n" + "\n".join(lines)


def retrieve_relevant_facts(query: str) -> list:
    """Returns up to TOP_K saved facts relevant to `query`, or [] if none
    are close enough. This is what keeps prompt size bounded as the number
    of saved facts grows — only the facts actually relevant to the current
    question get pulled in, not every fact ever saved."""
    facts = _load_facts()
    if not facts:
        return []

    try:
        collection = _get_memory_collection()
        query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
        results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    except Exception:
        return []

    relevant = []
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for doc, distance in zip(documents, distances):
        if distance <= DISTANCE_THRESHOLD:
            relevant.append(doc)
    return relevant


def get_relevant_facts_text(query: str) -> str:
    """The single entry point agent.py calls per message: returns the right
    facts text for this query — everything for a broad-recall question,
    only the relevant subset otherwise — or "" if nothing applies."""
    if is_broad_recall_query(query):
        return load_all_facts_text()

    relevant = retrieve_relevant_facts(query)
    if not relevant:
        return ""
    return "Known facts about the user (relevant to this question):\n" + "\n".join(f"- {f}" for f in relevant)