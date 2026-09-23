# Jarvis Agent

A personal AI agent, built incrementally, that can call tools, remember things
about me across sessions, answer from my own notes, and talk to me over
Telegram (and maybe voice next).

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
- [x] Phase 1 — core tool-calling loop (Ollama, local model: qwen2.5:3b-instruct)
- [x] Phase 2 — RAG over personal notes
- [x] Phase 3 — persistent memory (retrieval-based, with a deterministic
      path for broad "what do you know about me?" questions — see note below)
- [x] Phase 4 — Telegram interface
- [x] Phase 5a — voice IN (Telegram voice notes → text, via faster-whisper)
- [ ] Phase 5b — voice OUT (spoken replies)

## Setup

1. Install [Ollama](https://ollama.com) and pull the models this project uses:
   `ollama pull qwen2.5:3b-instruct` (chat) and `ollama pull nomic-embed-text` (embeddings)
2. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. Drop your own notes (`.txt` or `.md` files) into `notes/` — this folder
   is git-ignored, so your personal content never gets committed
5. `python src/rag/index_notes.py` — builds the searchable index (rerun this
   any time you add/edit/remove a note)
6. `python src/agent.py` — talk to Jarvis in the terminal. No `.env` needed
   for Phases 1-3.
7. For Telegram (Phase 4): copy `.env.example` to `.env`, fill in a bot
   token from @BotFather and your own numeric Telegram user ID (from
   @userinfobot — the bot only replies to this ID), then
   `python src/telegram_bot.py`. Note: this only works while the script is
   running on this machine — there's no cloud hosting here, so closing the
   laptop means Telegram messages go unanswered until it's running again.

## Known limitations (by design, not oversights)

- **General knowledge is unreliable for anything recent.** The local model's
  training has a cutoff; it'll answer confidently about well-known past
  facts but won't know about anything after its training data, and won't
  always flag that gap on its own. Not something local model tuning fixes —
  would need real web search grounding for that, a possible future step.
- **Occasional unnecessary tool calls.** A 3B model sometimes reaches for
  `remember_fact` or `get_current_datetime` on messages that don't need
  them. Harmless (deduped, or just an unused extra step) but not something
  prompt tuning fully eliminates at this model size.
- **Retrieval isn't perfectly precise with a small notes/memory corpus.**
  With only a handful of notes and facts saved, similarity search doesn't
  have much data to cleanly separate "relevant" from "irrelevant" — this
  improves naturally as more real notes and facts get added over time.

## Architecture

See each phase's section below as it's built.