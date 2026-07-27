import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from the environment variables or .env file.")

client = genai.Client(api_key=api_key)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    target_language: str = "Spanish"

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Language Tutor Bot</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f4f4f9; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
            .chat-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); width: 400px; display: flex; flex-direction: column; height: 500px; }
            #chat-box { flex: 1; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 4px; background: #fafafa; }
            .message { margin-bottom: 10px; padding: 8px; border-radius: 4px; }
            .user-msg { background: #d1e7dd; text-align: right; }
            .bot-msg { background: #cfe2ff; text-align: left; }
            .controls { display: flex; gap: 10px; margin-bottom: 10px; }
            input[type="text"] { flex: 1; padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
            button { padding: 8px 12px; background: #0d6efd; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0b5ed7; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <h2>Language Tutor Bot</h2>
            <div class="controls">
                <label for="lang">Target Lang:</label>
                <input type="text" id="lang" value="Spanish">
            </div>
            <div id="chat-box"></div>
            <div class="controls">
                <input type="text" id="user-input" placeholder="Type your message here..." onkeypress="if(event.key === 'Enter') sendMessage()">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            async function sendMessage() {
                const inputField = document.getElementById('user-input');
                const chatBox = document.getElementById('chat-box');
                const langField = document.getElementById('lang');
                
                const text = inputField.value.trim();
                const targetLanguage = langField.value.trim();
                if (!text) return;

                chatBox.innerHTML += `<div class="message user-msg"><b>You:</b> ${text}</div>`;
                inputField.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text, target_language: targetLanguage })
                    });
                    const data = await response.json();
                    
                    if (data.reply) {
                        chatBox.innerHTML += `<div class="message bot-msg"><b>Bot:</b> ${data.reply}</div>`;
                    } else {
                        chatBox.innerHTML += `<div class="message bot-msg" style="color:red;"><b>Error:</b> Could not get a response.</div>`;
                    }
                } catch (err) {
                    chatBox.innerHTML += `<div class="message bot-msg" style="color:red;"><b>Error:</b> Network connection failed.</div>`;
                }
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        prompt = (
            f"You are a helpful and encouraging language tutor. The user wants to practice {request.target_language}. "
            f"Reply to the user's message appropriately in {request.target_language}, but provide brief corrections "
            f"or English translations if they make a mistake, keeping the conversation engaging.\n\n"
            f"User message: {request.message}"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))