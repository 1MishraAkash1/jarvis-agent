"""
Phase 5b — text-to-speech (voice OUT).

Turns Jarvis's text reply into a spoken audio file. Mirrors Phase 5a's
transcribe.py in every way that matters:

- Local model, no cloud API, no per-request cost.
- Loaded once, lazily, on first real use.
- Picked specifically to avoid a PyTorch dependency: Piper runs on ONNX
  Runtime (already a dependency of faster-whisper, so this adds almost no
  extra install weight), same "avoid heavy deps" reasoning as Phase 2 and
  Phase 5a.

Unlike faster-whisper, Piper does NOT auto-download its model on first
use — you download a voice once with a separate command (see README), the
same one-time-setup shape as `ollama pull`. This module just loads
whatever's already on disk.

Piper produces a WAV file. Telegram voice-message replies need OGG/Opus,
so telegram_bot.py pipes this module's WAV output through ffmpeg to
convert it — see convert_wav_to_ogg() below.
"""

import os
import subprocess
import wave

from piper import PiperVoice

# Matches the voice you download with:
#   python3 -m piper.download_voices en_US-lessac-medium
# Change this if you download a different voice.
VOICE_NAME = "en_US-lessac-medium"

# Piper voices are two files: a .onnx model and a .onnx.json config,
# downloaded together into this directory (created if missing).
VOICE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "piper_voices")
MODEL_PATH = os.path.join(VOICE_DIR, f"{VOICE_NAME}.onnx")

_voice = None


def _get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Piper voice not found at {MODEL_PATH}. Download it once with:\n"
                f"  python3 -m piper.download_voices --download-dir "
                f"{VOICE_DIR} {VOICE_NAME}"
            )
        _voice = PiperVoice.load(MODEL_PATH)
    return _voice


def synthesize_speech(text: str, wav_path: str) -> None:
    """Synthesizes text to speech and writes it as a WAV file at wav_path."""
    voice = _get_voice()
    with wave.open(wav_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)


def convert_wav_to_ogg(wav_path: str, ogg_path: str) -> None:
    """Converts a WAV file to OGG/Opus — the format Telegram requires for a
    reply to display as a native playable voice-message bubble rather than
    a generic file attachment. Requires ffmpeg to be installed on the
    system (not a pip package — `sudo apt install ffmpeg` on Ubuntu)."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",  # overwrite output if it exists
            "-i", wav_path,
            "-c:a", "libopus",
            "-b:a", "32k",  # voice-quality bitrate is plenty for speech
            ogg_path,
        ],
        check=True,
        capture_output=True,
    )