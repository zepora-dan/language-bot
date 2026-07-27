# Language Practice Telegram Bot (Free Version)

A Telegram bot that chats with you to help practice a language of your choice --
by text or by voice -- using Google's **free** Gemini API. No billing/payment
required.

## How it works

1. `/start` -- pick a language from the menu (or "Other" to type your own),
   then pick your level (beginner / intermediate / advanced)
2. Chat normally by **typing** or by sending a **voice message**:
   - Typed messages get a text reply, plus a spoken audio reply (for
     supported languages)
   - Voice messages get transcribed by Gemini (so you can see how your
     speech came through), then get a text + spoken reply back
3. The bot gently corrects mistakes as part of the conversation
4. `/reset` -- start over with a new language or level

## Setup

### 1. Get a Telegram bot token
- Message [@BotFather](https://t.me/BotFather) on Telegram
- Send `/newbot` and follow the prompts
- Copy the token it gives you

### 2. Get a free Gemini API key
- Go to https://aistudio.google.com/app/apikey
- Sign in with a Google account
- Click **Create API key** -- no credit card needed for the free tier
- Copy the key

### 3. Install Python dependencies
```
pip install -r requirements.txt
```

### 4. Configure your keys
- Copy `.env.example` to `.env`
- Fill in your `TELEGRAM_BOT_TOKEN` and `GEMINI_API_KEY`

### 5. Run the bot
```
python bot.py
```

Open Telegram, find your bot, and send `/start`.

## Notes

- Conversation history is stored in memory only -- it resets if the bot restarts
- Chat replies and voice transcription both use `gemini-1.5-flash` (free tier)
- Spoken replies use `gTTS` (Google Translate's free text-to-speech) --
  supported for the listed menu languages; if a typed "Other" language isn't
  in the supported list, the bot still replies with text, just without audio
- No ffmpeg needed -- Gemini reads Telegram's voice format directly, and gTTS
  outputs a format Telegram accepts as-is
- Free tier has generous but not unlimited daily usage limits -- if you hit a
  quota error, wait a bit or check limits at https://ai.google.dev/pricing
- Good next steps: saving user preferences between restarts, adding more
  gTTS language codes, a pronunciation-scoring feature
