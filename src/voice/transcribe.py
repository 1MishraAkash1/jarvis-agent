"""
Phase 5a — speech-to-text (voice IN).

Converts a recorded audio file (a Telegram voice note, in our case) into
plain text, which then flows through the exact same process_message() the
text and Telegram front-ends already use — Jarvis doesn't know or care
whether a message started as typed text or a transcribed voice note.

We use faster-whisper rather than OpenAI's original openai-whisper package:
same underlying Whisper model, but faster-whisper runs on a CTranslate2
backend instead of PyTorch, which means a much lighter install (no ~2GB
PyTorch pull) and noticeably faster CPU transcription — the same
"avoid heavy dependencies when a lighter option does the same job"
reasoning we used back in Phase 2 (Ollama's own embeddings instead of
sentence-transformers/PyTorch).

The model file itself (~150MB for "base") downloads automatically from
Hugging Face the first time you transcribe anything, then is cached
locally — similar to how `ollama pull` works, just triggered automatically
on first use instead of a separate manual command.
"""

from faster_whisper import WhisperModel

# "base" is a reasonable speed/accuracy balance on CPU for short voice
# notes. "tiny" is faster but noticeably less accurate; "small"/"medium"
# are more accurate but slower — worth trying if "base" mistranscribes
# too often in practice.
MODEL_SIZE = "base"

# Loaded once, lazily, the first time transcribe_audio() is actually
# called — not at import time — so importing this module doesn't trigger
# a model download/load if voice features end up unused in some run.
_model = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        # int8 quantization on CPU: faster, small accuracy tradeoff — fine
        # for conversational voice notes, same tradeoff logic as the local
        # LLM quantization we've been running all along.
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe_audio(file_path: str) -> str:
    """Transcribes an audio file to text. Accepts most common audio
    formats (Telegram voice notes are .ogg/Opus) — faster-whisper decodes
    them internally, no manual format conversion needed."""
    model = _get_model()
    segments, _info = model.transcribe(file_path)
    return " ".join(segment.text.strip() for segment in segments).strip()