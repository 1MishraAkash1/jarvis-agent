# Jarvis Agent

A personal AI agent, built incrementally, that can call tools, remember things
about me across sessions, answer from my own notes, and eventually talk to me
via Telegram (and maybe voice).

Runs on a **local model via Ollama** — no API key, no per-request cost, no
cloud rate limits. (We started on Gemini's free tier; it was hitting
30-60+ second response times from server-side throttling, confirmed by
testing the raw API directly with curl — not a bug in this code. Moved to
Ollama running locally on an i9-13900H instead: 1-4 second responses,
consistently.)

Built by Akash Mishra as a learning project — agent architecture, tool-calling,
retrieval-augmented generation (RAG), and bot integration.

## Status

- [x] Phase 0 — project scaffold
- [x] Phase 1 — core tool-calling loop (Ollama, local model)
- [ ] Phase 2 — RAG over personal notes
- [ ] Phase 3 — persistent memory
- [ ] Phase 4 — Telegram interface
- [ ] Phase 5 — voice (stretch goal)

## Setup

1. Install [Ollama](https://ollama.com) and pull a tool-calling-capable model:
   `ollama pull llama3.1:8b`
2. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python src/agent.py` — no `.env` needed for Phase 1; that file is only
   used starting Phase 4 (Telegram bot token)

## Architecture

See each phase's section below as it's built.
