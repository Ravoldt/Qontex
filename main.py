import socket
import re
import threading
import google.generativeai as genai
import os
import time
import json
from dotenv import load_dotenv
from utils import Message, get_streamer_name, create_stream_folder, log_message, log_json, log_start_stop, shared_deque
import datetime
import streamlink as my_streamlink

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

streamer_name = CHANNEL.lstrip('#') if CHANNEL else "unknown"
stream_start = datetime.datetime.now()
log_folder = create_stream_folder(streamer_name, stream_start)
log_start_stop(log_folder, "start")

STREAM_START_TIME = None

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

def process_items_with_gemini():
    """Process the last 3 minutes of messages for collected items."""
    messages = shared_deque.get_recent()
    if not messages:
        return
    # Format as text
    context = "\n".join(str(msg) for msg in messages)
    prompt = f"""
    Based on this 3-minute context of transcript and chat from a Hollow Knight Silksong stream, list any items the streamer has collected.
    Compare against known Hollow Knight wiki items to remove hallucinations. Provide a list of confirmed collected items.
    Context:
    {context}
    """
    try:
        response = model.generate_content(prompt)
        items = response.text.strip()
        print(f"\n📦 Collected items: {items}\n")
        # Log to collected_items.json
        with open(os.path.join(log_folder, "collected_items.json"), "a", encoding='utf-8') as f:
            json.dump({"timestamp": datetime.datetime.now().isoformat(), "items": items}, f)
            f.write('\n')
        
        # We commented this out as recording.mp4 is no longer generated directly.
        # Video frames are now kept in an in-memory buffer via cv2.
        """
        recording_path = os.path.join(log_folder, "recording.mp4")
        if os.path.exists(recording_path) and messages:
            start_time = max(0, messages[0].timestamp)
            duration = 180  # 3 minutes
            output_path = os.path.join(log_folder, f"segment_{int(start_time)}.mp4")
            from utils import cut_video_segment
            cut_video_segment(recording_path, output_path, start_time, duration)
            print(f"Video segment saved to {output_path}")
        """
    except Exception as e:
        print(f"❌ Gemini item processing error: {e}")

def item_processor():
    """Background thread to process items every 30 seconds."""
    while True:
        time.sleep(30)
        process_items_with_gemini()

# --- BACKGROUND THREAD (Twitch Listener) ---
def irc_listener():
    global STREAM_START_TIME
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
                
                # Calculate timestamp relative to stream start
                msg_time = time.time() - STREAM_START_TIME
                msg = Message(msg_time, "chat", chat_msg, user=username)
                
                # DEBUG: Print every chat message to verify connection
                print(f"\r{msg}")
                print("Cmd (list, ask <#>, clear, quit)> ", end="", flush=True)

                log_message(log_folder, "chat.log", msg)
                log_json(log_folder, "merged.json", msg.to_dict())
                shared_deque.add_message(msg)

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
    global STREAM_START_TIME
    STREAM_START_TIME = time.time()
    
    # Expose the start time to the streamlink module so transcripts share exactly the same logic
    my_streamlink.STREAM_START_TIME = STREAM_START_TIME
    
    # Start the background listener
    listener_thread = threading.Thread(target=irc_listener, daemon=True)
    listener_thread.start()
    
    # Start the streamlink capture loop as a background thread
    source_channel = CHANNEL.lstrip('#') if CHANNEL else "babylon340"
    capture_thread = threading.Thread(target=my_streamlink.run_capture_loop, args=(source_channel, log_folder), daemon=True)
    capture_thread.start()
    
    # Start item processor
    processor_thread = threading.Thread(target=item_processor, daemon=True)
    processor_thread.start()
    
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
            uptime = time.time() - STREAM_START_TIME
            log_start_stop(log_folder, "stop", uptime=uptime)
            print("Shutting down...")
            break
