"""
Phase 4 — talk to Jarvis over Telegram.

This is a second front-end for the exact same brain as agent.py's terminal
version — same tool calling, same RAG over your notes, same memory. Nothing
about how Jarvis thinks changes here; this file only handles receiving
messages from Telegram and sending replies back.

IMPORTANT — read this before you rely on it: Ollama runs locally on THIS
machine. This script has to be running, on this machine, for Telegram
messages to get a reply. Close the laptop, kill this script, or lose
network — messages just won't be answered until it's running again. This
is not a hosted bot; your phone is a new way to talk to the same local
Jarvis, not a way to reach it from anywhere without your computer being on.

Security: a Telegram bot's username is effectively public — anyone who
finds it can message it. Since Jarvis can recall your personal notes and
saved facts through ordinary conversation, this bot only replies to ONE
Telegram user ID (yours, from TELEGRAM_ALLOWED_USER_ID in .env). Messages
from anyone else are silently ignored.
"""

import asyncio
import os
import tempfile

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from agent import new_conversation, process_message
from voice.transcribe import transcribe_audio
from voice.synthesize import convert_wav_to_ogg, synthesize_speech

load_dotenv()

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = os.environ.get("TELEGRAM_ALLOWED_USER_ID")

if not BOT_TOKEN or BOT_TOKEN == "your_telegram_bot_token_here":
    raise SystemExit(
        "No Telegram bot token found. Add TELEGRAM_BOT_TOKEN to your .env "
        "file (get one from @BotFather on Telegram)."
    )
if not ALLOWED_USER_ID:
    raise SystemExit(
        "No TELEGRAM_ALLOWED_USER_ID set in .env. Message @userinfobot on "
        "Telegram to get your numeric user ID, then add it to .env — "
        "without this, anyone who finds your bot could talk to it and pull "
        "out your personal notes/facts through conversation."
    )
ALLOWED_USER_ID = int(ALLOWED_USER_ID)

# One shared conversation history for the bot process. This is a personal,
# single-user assistant (locked to one Telegram ID above), so — like the
# terminal version — we don't need per-user session tracking. History
# resets when this script restarts; your actual notes and saved facts
# don't (those live on disk in data/ and notes/, untouched by this).
messages = new_conversation()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id != ALLOWED_USER_ID:
        # Silently ignore — don't reveal that this is a private bot, and
        # definitely don't let a stranger's message reach the model at all.
        print(f"  [ignored message from unauthorized user id={user_id}]")
        return

    user_input = update.message.text.strip()
    if not user_input:
        return

    print(f"You (Telegram): {user_input}")

    # ollama.chat() is a blocking (synchronous) call. Running it directly
    # inside this async handler would freeze the bot's event loop for the
    # whole 1-2+ seconds it takes to respond — fine with one user, but the
    # correct pattern either way is to run blocking work in a thread so the
    # bot can still process Telegram's own housekeeping messages meanwhile.
    reply_text = await asyncio.to_thread(process_message, user_input, messages)

    print(f"Jarvis: {reply_text}\n")
    await update.message.reply_text(reply_text)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id != ALLOWED_USER_ID:
        print(f"  [ignored voice note from unauthorized user id={user_id}]")
        return

    print("You (Telegram): [voice note received, transcribing...]")

    # Telegram voice notes are stored server-side; get_file() gives us a
    # short-lived download handle, then we pull the actual bytes down to a
    # temp file. Whisper needs a real file path, not raw bytes in memory,
    # so this temp file is just a hand-off point.
    voice_file = await context.bot.get_file(update.message.voice.file_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        await voice_file.download_to_drive(tmp_path)

        # Both the download-to-disk step above and the transcription below
        # are blocking I/O/CPU work — same reasoning as process_message()
        # already gets in handle_message(): run it off the event loop.
        transcript = await asyncio.to_thread(transcribe_audio, tmp_path)
    finally:
        # Always clean up, even if transcription raised — this is a temp
        # file with someone's spoken words in it, no reason to let it pile
        # up on disk.
        os.remove(tmp_path)

    transcript = transcript.strip()
    if not transcript:
        print("Jarvis: [could not make out any speech in that voice note]\n")
        await update.message.reply_text(
            "I couldn't make out any speech in that voice note — could you try again?"
        )
        return

    print(f"You (Telegram, transcribed): {transcript}")

    reply_text = await asyncio.to_thread(process_message, transcript, messages)

    print(f"Jarvis: {reply_text}\n")

    # Echo the transcript back first so you can actually tell, especially
    # on this first test, whether Whisper heard you correctly — if the
    # reply seems off, this line is how you'll know it's an STT mistake
    # and not Jarvis misunderstanding a correctly-heard message.
    await update.message.reply_text(f"Heard: \"{transcript}\"\n\n{reply_text}")
    
    # Phase 5b: since you spoke to Jarvis, speak the reply back too. Text
    # messages still get text-only replies (handle_message never calls
    # this) — voice in gets voice out, typing gets typing back. The text
    # reply above is sent regardless, so a TTS failure here never costs
    # you the actual answer.
    await send_voice_reply(update, reply_text)

async def send_voice_reply(update: Update, text: str) -> None:
    """Synthesizes text to speech and sends it back as a Telegram voice
    message. Best-effort: if Piper's voice model isn't downloaded yet, or
    ffmpeg isn't installed, this logs a warning and does nothing further —
    it never raises, since the text reply was already sent and is the
    part that actually matters."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
        wav_path = wav_tmp.name
    ogg_path = wav_path.replace(".wav", ".ogg")

    try:
        await asyncio.to_thread(synthesize_speech, text, wav_path)
        await asyncio.to_thread(convert_wav_to_ogg, wav_path, ogg_path)
        with open(ogg_path, "rb") as ogg_file:
            await update.message.reply_voice(ogg_file)
    except Exception as exc:  # noqa: BLE001 — deliberately broad, see docstring
        print(f"  [voice reply failed, sent text only: {exc}]")
    finally:
        for path in (wav_path, ogg_path):
            if os.path.exists(path):
                os.remove(path)
                
def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("Jarvis Telegram bot running. Message your bot on Telegram to talk to it.")
    print("Press Ctrl+C to stop.\n")
    app.run_polling()


if __name__ == "__main__":
    main()