import socket
import re
import threading
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()  # Loads variables from the local .env file

api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("CRITICAL ERROR: GENAI_API_KEY is missing! Please check your .env file.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-3-flash-preview')

NICK = os.getenv("TWITCH_USERNAME")
PASS = os.getenv("TWITCH_TOKEN")
CHANNEL = os.getenv("CHANNEL")
if CHANNEL and not CHANNEL.startswith("#"):
    CHANNEL = f"#{CHANNEL}"

# The Queue
question_queue = []

# --- LOCAL SCREENING ---
def is_likely_question(message):
    message = message.lower().strip()
    if message.endswith("?"): return True
    question_starters = ["who", "what", "where", "when", "why", "how", "can you", "is there", "do you", "does it", "are there", "could you", "would you", "should i", "will it"]
    if any(message.startswith(q) for q in question_starters): return True
    return False

# --- GEMINI MODULE ---
def ask_gemini(username, question):
    system_prompt = f"""
    The user '{username}' asked this question: "{question}"
    Provide a concise, factual answer. Do not format for chat, this is for my personal reading.
    """
    try:
        print("\n[Thinking...]")
        response = model.generate_content(system_prompt)
        print(f"\n💡 Answer for {username}:\n{response.text.strip()}\n")
    except Exception as e:
        print(f"\n❌ Gemini API Error: {e}\n")

# --- BACKGROUND THREAD (Twitch Listener) ---
def irc_listener():
    sock = socket.socket()
    sock.connect(("irc.chat.twitch.tv", 6667)) # Using 6667 for simplicity in this example
    sock.send(f"PASS {PASS}\r\n".encode("utf-8"))
    sock.send(f"NICK {NICK}\r\n".encode("utf-8"))
    sock.send(f"JOIN {CHANNEL}\r\n".encode("utf-8"))

    while True:
        try:
            response = sock.recv(2048).decode("utf-8")
            if response.startswith("PING"):
                sock.send("PONG :tmi.twitch.tv\r\n".encode("utf-8"))
            
            match = re.search(r":(\w+)!\w+@\w+\.tmi\.twitch\.tv PRIVMSG #\w+ :(.*)", response)
            if match:
                username = match.group(1)
                chat_msg = match.group(2).strip()
                
                # DEBUG: Print every chat message to verify connection
                print(f"\r[DEBUG CHAT] {username}: {chat_msg}")
                print("Cmd (list, ask <#>, clear, quit)> ", end="", flush=True)

                if is_likely_question(chat_msg):
                    question_queue.append({"user": username, "msg": chat_msg})
                    # Use \r to overwrite the input prompt briefly so it looks clean
                    print(f"\r[!] New question detected! (Total in queue: {len(question_queue)})")
                    print("Cmd (list, ask <#>, clear, quit)> ", end="", flush=True)
                    
        except Exception as e:
            print(f"\r[!] Listener Error: {e}")
            print("Cmd (list, ask <#>, clear, quit)> ", end="", flush=True)

# --- MAIN THREAD (Your CLI Interface) ---
if __name__ == "__main__":
    # Start the background listener
    listener_thread = threading.Thread(target=irc_listener, daemon=True)
    listener_thread.start()
    
    print(f"Connected to {CHANNEL}. Listening for questions in the background...")
    
    while True:
        # This keeps running without blocking the IRC connection
        cmd = input("Cmd (list, ask <#>, clear, quit)> ").strip().lower()
        
        if cmd == "list":
            print("\n--- Question Queue ---")
            if not question_queue:
                print("Queue is empty.")
            for i, q in enumerate(question_queue):
                print(f"[{i}] {q['user']}: {q['msg']}")
            print("----------------------\n")
            
        elif cmd.startswith("ask "):
            try:
                idx = int(cmd.split(" ")[1])
                q = question_queue.pop(idx) # Removes it from the queue and selects it
                ask_gemini(q['user'], q['msg'])
            except (IndexError, ValueError):
                print("Invalid ID. Type 'list' to see IDs, then 'ask 0' for example.\n")
                
        elif cmd == "clear":
            question_queue.clear()
            print("Queue cleared.\n")
            
        elif cmd == "quit":
            print("Shutting down...")
            break