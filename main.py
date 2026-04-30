import threading
import os
import time
import json
import datetime
import argparse
from dotenv import load_dotenv

from utils import get_streamer_name, create_stream_folder, log_start_stop, refresh_twitch_token
from twitch_chat import TwitchChatListener
from gemini_agent import GeminiAgent
import streamlink as my_streamlink

def main():
    parser = argparse.ArgumentParser(description="Qontex Stream AI Agent")
    parser.add_argument("--test-chat", action="store_true", help="Run only the Twitch Chat module")
    parser.add_argument("--test-capture", action="store_true", help="Run only the Video/Audio Capture module")
    parser.add_argument("--no-gemini", action="store_true", help="Run without sending context/questions to Gemini")
    args = parser.parse_args()

    load_dotenv()
    refresh_twitch_token()
    
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key and not args.no_gemini:
        print("CRITICAL ERROR: GENAI_API_KEY is missing! Please check your .env file.")
        exit(1)

    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("CRITICAL ERROR: config.json is missing!")
        exit(1)

    CHANNEL = config.get("CHANNEL")
    TWITCH_USERNAME = config.get("TWITCH_USERNAME")
    PROCESS_FAST = config.get("PROCESS_FAST", False)
    ENABLE_QA = config.get("ENABLE_QA", True)
    ENABLE_ITEMS = config.get("ENABLE_ITEMS", True)

    if not CHANNEL:
        print("Please set the CHANNEL in config.json!")
        exit(1)

    is_local_video = os.path.isfile(CHANNEL)
    if not is_local_video:
        if not CHANNEL.startswith("#"):
            CHANNEL = f"#{CHANNEL}"
        source_channel = CHANNEL.lstrip('#')
    else:
        source_channel = CHANNEL
    GAME_NAME = config.get("GAME_NAME") or config.get("GAME") or source_channel
    
    stream_start = datetime.datetime.now()
    log_folder = create_stream_folder(source_channel, stream_start)
    log_start_stop(log_folder, "start")

    # Time reference for all modules
    STREAM_START_TIME = time.time()
    def get_msg_time():
        return time.time() - STREAM_START_TIME

    def answer_question(msg):
        if agent:
            agent.ask_gemini(msg.user, msg.text, source_type=msg.type, timestamp=msg.timestamp)

    def update_stream_category(category):
        if agent:
            agent.set_game_name(category)

    # If running specific tests
    if args.test_chat:
        if is_local_video:
            print("Cannot test Twitch chat with a local video file.")
            return
        print(f"Running standalone Twitch Chat test for {CHANNEL}")
        PASS = os.getenv("TWITCH_TOKEN")
        listener = TwitchChatListener(
            TWITCH_USERNAME,
            PASS,
            CHANNEL,
            log_folder,
            start_time_ref=get_msg_time,
            category_handler=lambda category: print(f"Using Twitch category: {category}"),
        )
        listener.listen()
        return

    if args.test_capture:
        print(f"Running standalone Capture test for {source_channel}")
        my_streamlink.STREAM_START_TIME = STREAM_START_TIME
        my_streamlink.run_capture_loop(source_channel, log_folder, process_fast=PROCESS_FAST)
        return

    # Normal execution:
    my_streamlink.STREAM_START_TIME = STREAM_START_TIME
    
    # 1. Start Agent Processor
    agent = None
    if not args.no_gemini:
        agent = GeminiAgent(api_key, log_folder, start_time_ref=get_msg_time, game_name=GAME_NAME or source_channel)
        if ENABLE_ITEMS:
            def item_processor():
                while True:
                    time.sleep(30)
                    agent.process_items(video_frames_deque=my_streamlink.video_frames)
                    
            processor_thread = threading.Thread(target=item_processor, daemon=True)
            processor_thread.start()
        else:
            print("Gemini Item Processing is DISABLED via config.json.")
    else:
        print("Gemini Agent is DISABLED (--no-gemini flag used).")

    qa_handler = answer_question if (agent and ENABLE_QA) else None

    # 2. Start Chat Listener
    PASS = os.getenv("TWITCH_TOKEN")
    chat_listener = None
    if not is_local_video:
        chat_listener = TwitchChatListener(
            TWITCH_USERNAME,
            PASS,
            CHANNEL,
            log_folder,
            start_time_ref=get_msg_time,
            question_handler=qa_handler,
            category_handler=update_stream_category,
        )
        listener_thread = threading.Thread(target=chat_listener.listen, daemon=True)
        listener_thread.start()
        print(f"Connected to {CHANNEL}. Listening for questions in the background...")
    else:
        print(f"Local video '{CHANNEL}' detected. Twitch chat listener is disabled.")
    
    # 3. Start Capture Loop
    capture_thread = threading.Thread(
        target=my_streamlink.run_capture_loop,
        args=(source_channel, log_folder, PROCESS_FAST, qa_handler),
        daemon=True,
    )
    capture_thread.start()

    
    while True:
        cmd = input("Cmd (status, clear, quit)> ").strip().lower()
        
        if cmd in ("status", "list"):
            print("\n--- Status ---")
            if not chat_listener or not chat_listener.question_queue:
                print("Question queue is empty.")
            else:
                print("Queued questions:")
                for i, q in enumerate(chat_listener.question_queue):
                    print(f"[{i}] {q['user']}: {q['msg']}")
            print("--------------\n")
                
        elif cmd == "clear":
            if chat_listener:
                chat_listener.question_queue.clear()
            print("Queue cleared.\n")
            
        elif cmd == "quit":
            uptime = time.time() - STREAM_START_TIME
            log_start_stop(log_folder, "stop", uptime=uptime)
            print("Shutting down...")
            break

if __name__ == "__main__":
    main()
