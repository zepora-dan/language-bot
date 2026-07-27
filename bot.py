import os
import logging
from io import BytesIO

from dotenv import load_dotenv
import google.generativeai as genai
from gtts import gTTS

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
    raise RuntimeError(
        "Missing TELEGRAM_BOT_TOKEN or GEMINI_API_KEY. "
        "Add them to a .env file (see .env.example)."
    )

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3.5-flash-lite"  # current free-tier model (2.5-flash no longer available to new users)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Language & level menus
# ---------------------------------------------------------
LANGUAGES = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese",
    "Japanese", "Korean", "Mandarin Chinese", "Arabic",
    "Russian", "Hindi", "Amharic", "Other (type it)",
]

LEVELS = ["Beginner", "Intermediate", "Advanced"]

# Maps a language name to the code gTTS needs for spoken replies.
# If a language isn't listed here (e.g. a custom "Other" language,
# or one gTTS doesn't support), spoken replies are skipped gracefully
# and the bot still replies with text.
GTTS_LANG_CODES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "japanese": "ja",
    "korean": "ko",
    "mandarin chinese": "zh-CN",
    "arabic": "ar",
    "russian": "ru",
    "hindi": "hi",
    "amharic": "am",
}


def chunk(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


LANGUAGE_KEYBOARD = ReplyKeyboardMarkup(
    chunk(LANGUAGES, 2), one_time_keyboard=True, resize_keyboard=True
)
LEVEL_KEYBOARD = ReplyKeyboardMarkup(
    chunk(LEVELS, 3), one_time_keyboard=True, resize_keyboard=True
)

# ---------------------------------------------------------
# In-memory per-user state
# ---------------------------------------------------------
user_sessions: dict[int, dict] = {}

MAX_HISTORY_MESSAGES = 10  # keep the last N turns to limit token usage


def build_system_prompt(known_language: str, target_language: str, level: str) -> str:
    return (
        f"You are a friendly, patient conversation partner helping someone practice "
        f"{target_language}. They already know {known_language}, and their self-rated "
        f"level in {target_language} is {level}. "
        f"Reply mainly in {target_language} to keep the conversation flowing. "
        f"If they make a grammar or vocabulary mistake, gently point it out and give the "
        f"corrected version -- explain the correction in {known_language} so it's clear, "
        f"then continue the conversation in {target_language}. "
        f"Keep replies conversational, short, and clear -- a sentence or two at most, "
        f"since replies are also read aloud and long replies are harder to follow as speech. "
        f"If their level is beginner, use simpler vocabulary and shorter sentences, and "
        f"feel free to add a short {known_language} translation after new or difficult "
        f"words or phrases. "
        f"Special case: if their message is just a single word (in either language), do "
        f"not reply conversationally -- reply with ONLY that word's translation between "
        f"{known_language} and {target_language} (translate to whichever language it "
        f"isn't already in), with nothing else added."
    )


def get_gtts_lang_code(language: str) -> str | None:
    return GTTS_LANG_CODES.get(language.strip().lower())


# ---------------------------------------------------------
# Handlers
# ---------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_sessions[user_id] = {"stage": "awaiting_known_language"}
    await update.message.reply_text(
        "Hi! I'm your language practice buddy. \U0001F5E3\uFE0F\n\n"
        "Which language do you already know well? Pick one below, "
        "or choose 'Other (type it)' to enter your own.",
        reply_markup=LANGUAGE_KEYBOARD,
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_sessions[user_id] = {"stage": "awaiting_known_language"}
    await update.message.reply_text(
        "Okay, let's start over. Which language do you already know well?",
        reply_markup=LANGUAGE_KEYBOARD,
    )


async def generate_reply(session: dict, user_text: str) -> str:
    """Send the conversation so far to Gemini and return the reply text."""
    history = session["history"]
    history.append({"role": "user", "parts": [user_text]})
    trimmed_history = history[-MAX_HISTORY_MESSAGES:]

    system_prompt = build_system_prompt(
        session["known_language"], session["language"], session["level"]
    )
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        system_instruction=system_prompt,
    )
    response = model.generate_content(trimmed_history)
    reply = response.text

    history.append({"role": "model", "parts": [reply]})
    session["history"] = history
    return reply


async def transcribe_audio(ogg_bytes: bytes) -> str:
    """Use Gemini's audio understanding to transcribe a voice message."""
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content([
        {"mime_type": "audio/ogg", "data": ogg_bytes},
        "Transcribe exactly what is said in this audio. "
        "Output only the transcription, nothing else.",
    ])
    return response.text.strip()


def synthesize_speech(text: str, language: str) -> BytesIO | None:
    """Convert text to speech using gTTS (free, no API key needed).
    Returns None if the language isn't supported by gTTS."""
    lang_code = get_gtts_lang_code(language)
    if lang_code is None:
        return None

    mp3_bytes = BytesIO()
    tts = gTTS(text=text, lang=lang_code)
    tts.write_to_fp(mp3_bytes)
    mp3_bytes.seek(0)
    mp3_bytes.name = "reply.mp3"
    return mp3_bytes


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    text = update.message.text.strip()

    session = user_sessions.get(user_id)

    if session is None:
        await update.message.reply_text("Send /start to begin practicing a language!")
        return

    # Step 1: waiting for the language they already know
    if session["stage"] == "awaiting_known_language":
        if text == "Other (type it)":
            await update.message.reply_text(
                "No problem -- type the name of the language you already know well.",
                reply_markup=ReplyKeyboardRemove(),
            )
            return
        session["known_language"] = text
        session["stage"] = "awaiting_target_language"
        await update.message.reply_text(
            f"Got it, {text}. Now, which language would you like to practice? "
            f"Pick one below, or choose 'Other (type it)' to enter your own.",
            reply_markup=LANGUAGE_KEYBOARD,
        )
        return

    # Step 2: waiting for the language they want to learn/practice
    if session["stage"] == "awaiting_target_language":
        if text == "Other (type it)":
            await update.message.reply_text(
                "No problem -- type the name of the language you'd like to practice.",
                reply_markup=ReplyKeyboardRemove(),
            )
            return
        session["language"] = text
        session["stage"] = "awaiting_level"
        await update.message.reply_text(
            f"Great, {text} it is! What's your level?",
            reply_markup=LEVEL_KEYBOARD,
        )
        return

    # Step 3: waiting for level choice
    if session["stage"] == "awaiting_level":
        session["level"] = text
        session["stage"] = "chatting"
        session["history"] = []
        await update.message.reply_text(
            f"Perfect. Let's start chatting in {session['language']} "
            f"({session['level']} level) -- I'll explain things in "
            f"{session['known_language']} when helpful. You can type or send a "
            f"voice message. I'll reply with text, and with audio too when "
            f"available for this language. Send /reset any time to change "
            f"language or level.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return

    # Step 4: normal conversation practice (text)
    if session["stage"] == "chatting":
        try:
            reply = await generate_reply(session, text)
        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
            await update.message.reply_text(
                "Sorry, I had trouble generating a reply just now. Please try again."
            )
            return

        await update.message.reply_text(reply)

        try:
            audio = synthesize_speech(reply, session["language"])
            if audio is not None:
                await update.message.reply_audio(audio=audio)
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            # Text reply already sent, so a TTS failure isn't fatal -- just skip audio.


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    session = user_sessions.get(user_id)

    if session is None or session.get("stage") != "chatting":
        await update.message.reply_text(
            "Send /start first and pick a language/level before sending voice messages."
        )
        return

    # Download the voice message (Telegram sends OGG/Opus -- Gemini can read this directly)
    voice_file = await update.message.voice.get_file()
    ogg_bytes = BytesIO()
    await voice_file.download_to_memory(out=ogg_bytes)
    ogg_bytes.seek(0)

    try:
        user_text = await transcribe_audio(ogg_bytes.read())
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        await update.message.reply_text(
            "Sorry, I couldn't understand that audio. Could you try again or type it instead?"
        )
        return

    await update.message.reply_text(f"I heard: \u201c{user_text}\u201d")

    try:
        reply = await generate_reply(session, user_text)
    except Exception as e:
        logger.error(f"Gemini chat error: {e}")
        await update.message.reply_text(
            "Sorry, I had trouble generating a reply just now. Please try again."
        )
        return

    await update.message.reply_text(reply)

    try:
        audio = synthesize_speech(reply, session["language"])
        if audio is not None:
            await update.message.reply_audio(audio=audio)
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
