import telebot
import requests
import sys
import os
from typing import Dict, List
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from talk_character.constants import SYSTEM_PROMPT_RICK

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
MLX_SERVER_URL = "http://localhost:8080/v1/chat/completions"

bot = telebot.TeleBot(BOT_TOKEN)
chat_history: Dict[int, List[dict]] = {}

SYSTEM_PROMPT = {
    "role": "system",
    "content": SYSTEM_PROMPT_RICK
}

@bot.message_handler(commands=['start', 'reset'])
def start_chat(message):
    chat_id = message.chat.id
    chat_history[chat_id] = [SYSTEM_PROMPT]
    bot.reply_to(message, "Hi, I'm Rick.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    chat_id = message.chat.id
    user_text = message.text
    
    if chat_id not in chat_history:
        chat_history[chat_id] = [SYSTEM_PROMPT]
    
    chat_history[chat_id].append({"role": "user", "content": user_text})
    
    payload = {
        "messages": chat_history[chat_id],
        "temperature": 0.7,
        "max_tokens": 200
    }
    
    try:
        response = requests.post(MLX_SERVER_URL, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        ai_response = response.json()["choices"][0]["message"]["content"]
        chat_history[chat_id].append({"role": "assistant", "content": ai_response})
        
        bot.reply_to(message, ai_response)
        
    except requests.exceptions.RequestException as e:
        bot.reply_to(message, "🔧 Server's down. Try again later.")
        print(f"Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Rick bot...")
    bot.infinity_polling() 