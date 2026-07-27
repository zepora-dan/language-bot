import os
import re
import asyncio
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

# The model is asked to always separate its reply into two parts using this
# marker: everything before it is pure target-language content (safe to read
# aloud in the target language's voice), everything after is the
# known-language explanation (never spoken aloud).
REPLY_DELIMITER = "@@@"


def build_system_prompt(known_language: str, target_language: str, level: str) -> str:
    return (
        f"You are a translation and grammar-correction tool, NOT a conversation partner. "
        f"The user already knows {known_language} and is learning {target_language} "
        f"(self-rated level: {level}). You have exactly two jobs, and nothing else:\n\n"
        f"1) If the user writes in {known_language} (or any other language): translate "
        f"what they wrote into {target_language}.\n\n"
        f"2) If the user writes in {target_language}: check it carefully for grammar "
        f"mistakes (verb conjugation, word order, agreement, tense, articles, etc.) and "
        f"vocabulary mistakes, and give the corrected version in {target_language}. If "
        f"there is no mistake, just restate it correctly in {target_language}.\n\n"
        f"STRICT OUTPUT FORMAT -- this is critical because the first part is converted "
        f"to speech audio, so it must NEVER contain any {known_language} text:\n"
        f"  <content entirely in {target_language} -- the translation or corrected/"
        f"confirmed sentence, nothing else, no {known_language} words mixed in>\n"
        f"  {REPLY_DELIMITER}\n"
        f"  <short explanation entirely in {known_language} of any correction made, or "
        f"leave this part empty if there was no mistake to explain>\n\n"
        f"Always include the '{REPLY_DELIMITER}' marker on its own line exactly once, "
        f"even if the explanation part is empty. Do NOT put any {known_language} before "
        f"the marker, and do NOT put any {target_language} explanation after it beyond "
        f"what's naturally part of the explanation.\n\n"
        f"Do NOT hold a conversation. Do NOT answer questions, give opinions, share facts, "
        f"or respond to the meaning/content of what the user writes -- treat every message "
        f"purely as text to translate or correct, never as something to respond to "
        f"conversationally. If the user writes a question (in either language), do not "
        f"answer the question itself -- only translate it or correct its grammar. This "
        f"applies no matter what the message is about, including anything unrelated to "
        f"{target_language} -- you are not a general assistant, only a translation/"
        f"correction tool for {target_language}. "
        f"Keep the {target_language} part short and clear, since it is read aloud as "
        f"audio. If the level is beginner, use simpler vocabulary where natural."
    )


def get_gtts_lang_code(language: str) -> str | None:
    return GTTS_LANG_CODES.get(language.strip().lower())


def clean_for_tts(text: str) -> str:
    """Strip punctuation so gTTS doesn't pronounce it aloud.
    The original text (with punctuation) is still used for the on-screen
    text reply -- only the audio version goes through this."""
    cleaned = re.sub(r'[.,!?;:"\'()\[\]{}\-\u2013\u2014\u201c\u201d\u2018\u2019]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


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


async def generate_reply(session: dict, user_text: str) -> tuple[str, str]:
    """Send the conversation so far to Gemini and return (display_text, audio_text).
    display_text is shown as the chat message (target-language content plus, if
    present, the known-language explanation). audio_text is ONLY the
    target-language portion -- this is what gets converted to speech, so the
    user never hears known-language text read in the wrong voice."""
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
    # generate_content is a blocking network call -- run it off the event loop
    # so one user's reply being generated doesn't freeze the bot for everyone else.
    response = await asyncio.to_thread(model.generate_content, trimmed_history)
    raw_reply = response.text

    history.append({"role": "model", "parts": [raw_reply]})
    session["history"] = history

    if REPLY_DELIMITER in raw_reply:
        target_part, _, explanation_part = raw_reply.partition(REPLY_DELIMITER)
    else:
        # Model didn't follow the format -- treat the whole thing as target-language
        # content so at least the audio still isn't mixed-language.
        target_part, explanation_part = raw_reply, ""

    audio_text = target_part.strip()
    explanation_part = explanation_part.strip()
    display_text = audio_text if not explanation_part else f"{audio_text}\n\n{explanation_part}"

    return display_text, audio_text


async def transcribe_audio(ogg_bytes: bytes) -> str:
    """Use Gemini's audio understanding to transcribe a voice message."""
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = await asyncio.to_thread(
        model.generate_content,
        [
            {"mime_type": "audio/ogg", "data": ogg_bytes},
            "Transcribe exactly what is said in this audio. "
            "Output only the transcription, nothing else.",
        ],
    )
    return response.text.strip()


def synthesize_speech(text: str, language: str) -> BytesIO | None:
    """Convert text to speech using gTTS (free, no API key needed).
    Returns None if the language isn't supported by gTTS."""
    lang_code = get_gtts_lang_code(language)
    if lang_code is None:
        return None

    tts_text = clean_for_tts(text)
    if not tts_text:
        return None

    mp3_bytes = BytesIO()
    tts = gTTS(text=tts_text, lang=lang_code)
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
            f"Perfect. Send me anything -- a word, sentence, or paragraph -- and here's "
            f"what I'll do:\n"
            f"- If you write in {session['known_language']}, I'll translate it into "
            f"{session['language']}.\n"
            f"- If you write in {session['language']}, I'll check your grammar and "
            f"correct any mistakes (explained in {session['known_language']}).\n"
            f"I won't chat or answer questions -- just translate and correct. You can "
            f"type or send a voice message. Send /reset any time to change language or "
            f"level.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return

    # Step 4: normal conversation practice (text)
    if session["stage"] == "chatting":
        try:
            display_text, audio_text = await generate_reply(session, text)
        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
            await update.message.reply_text(
                "Sorry, I had trouble generating a reply just now. Please try again."
            )
            return

        await update.message.reply_text(display_text)

        try:
            audio = await asyncio.to_thread(synthesize_speech, audio_text, session["language"])
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
        display_text, audio_text = await generate_reply(session, user_text)
    except Exception as e:
        logger.error(f"Gemini chat error: {e}")
        await update.message.reply_text(
            "Sorry, I had trouble generating a reply just now. Please try again."
        )
        return

    await update.message.reply_text(display_text)

    try:
        audio = await asyncio.to_thread(synthesize_speech, audio_text, session["language"])
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
