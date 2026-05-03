import threading
import os
import time
import datetime
import argparse
from dotenv import load_dotenv

from utils import create_stream_folder, load_config, log_start_stop, refresh_twitch_token, reload_config, shared_deque
from twitch_chat import TwitchChatListener
from gemini_agent import GeminiAgent
import streamlink as my_streamlink

def resolve_config(config):
    channel = config.get("CHANNEL")
    if not channel:
        raise ValueError("Please set the CHANNEL in config.json!")

    is_local_video = os.path.isfile(channel)
    if not is_local_video:
        twitch_channel = channel if channel.startswith("#") else f"#{channel}"
        source = twitch_channel.lstrip("#")
    else:
        twitch_channel = channel
        source = channel

    return {
        "channel": twitch_channel,
        "source": source,
        "is_local_video": is_local_video,
        "stream_name": source,
        "game_name": config.get("GAME_NAME") or config.get("GAME") or source,
        "twitch_username": config.get("TWITCH_USERNAME"),
        "process_fast": config.get("PROCESS_FAST", False),
        "enable_qa": config.get("ENABLE_QA", True),
        "enable_items": config.get("ENABLE_ITEMS", True),
        "enable_visual_context": config.get("ENABLE_VISUAL_CONTEXT", False),
        "qa_context_window": config.get("QA_CONTEXT_WINDOW", 60),
    }

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
        config = load_config()
        runtime_config = resolve_config(config)
    except FileNotFoundError:
        print("CRITICAL ERROR: config.json is missing!")
        exit(1)
    except ValueError as e:
        print(e)
        exit(1)

    stream_start = datetime.datetime.now()
    log_folder = create_stream_folder(runtime_config["stream_name"], stream_start)
    log_start_stop(log_folder, "start")

    # Time reference for all modules
    STREAM_START_TIME = time.time()
    def get_msg_time():
        return time.time() - STREAM_START_TIME

    def update_stream_start(ts):
        nonlocal STREAM_START_TIME
        STREAM_START_TIME = ts
        my_streamlink.STREAM_START_TIME = ts

    def answer_question(msg):
        if agent:
            agent.ask_gemini(
                msg.user,
                msg.text,
                source_type=msg.type,
                timestamp=msg.timestamp,
                video_frames_deque=my_streamlink.video_frames,
            )

    def update_stream_category(category):
        if agent:
            agent.set_game_name(category)

    print("\nPreloading AI models into VRAM... (This will pause the script until ready)")
    if not args.test_chat:
        my_streamlink.preload_models()
        
    if not args.test_capture:
        from utils import preload_classifier
        preload_classifier()
    print("All models successfully loaded!\n")

    # If running specific tests
    if args.test_chat:
        if runtime_config["is_local_video"]:
            print("Cannot test Twitch chat with a local video file.")
            return
        print(f"Running standalone Twitch Chat test for {runtime_config['channel']}")
        PASS = os.getenv("TWITCH_TOKEN")
        listener = TwitchChatListener(
            runtime_config["twitch_username"],
            PASS,
            runtime_config["channel"],
            log_folder,
            start_time_ref=get_msg_time,
            category_handler=lambda category: print(f"Using Twitch category: {category}"),
            stream_start_handler=update_stream_start,
        )
        listener.listen()
        return

    if args.test_capture:
        print(f"Running standalone Capture test for {runtime_config['source']}")
        my_streamlink.STREAM_START_TIME = STREAM_START_TIME
        my_streamlink.run_capture_loop(
            runtime_config["source"],
            log_folder,
            process_fast=runtime_config["process_fast"],
        )
        return

    # Normal execution:
    my_streamlink.STREAM_START_TIME = STREAM_START_TIME

    runtime = {
        "config": runtime_config,
        "log_folder": log_folder,
        "chat_listener": None,
        "listener_thread": None,
        "capture_thread": None,
        "capture_stop": None,
        "enable_items": runtime_config["enable_items"],
    }

    agent = None
    if not args.no_gemini:
        agent = GeminiAgent(
            api_key,
            log_folder,
            start_time_ref=get_msg_time,
            game_name=runtime_config["game_name"],
            qa_context_window=runtime_config["qa_context_window"],
            enable_visual_context=runtime_config["enable_visual_context"],
        )

        def item_processor():
            while True:
                time.sleep(30)
                if runtime["enable_items"]:
                    agent.process_items(video_frames_deque=my_streamlink.video_frames)

        processor_thread = threading.Thread(target=item_processor, daemon=True)
        processor_thread.start()
        if not runtime_config["enable_items"]:
            print("Gemini Item Processing is DISABLED via config.json.")
    else:
        print("Gemini Agent is DISABLED (--no-gemini flag used).")

    def current_qa_handler(config):
        return answer_question if (agent and config["enable_qa"]) else None

    def stop_runtime():
        if runtime["chat_listener"]:
            runtime["chat_listener"].stop()
            runtime["chat_listener"] = None

        if runtime["capture_stop"]:
            runtime["capture_stop"].set()

        if runtime["capture_thread"] and runtime["capture_thread"].is_alive():
            runtime["capture_thread"].join(timeout=5)

        runtime["capture_thread"] = None
        runtime["capture_stop"] = None

    def start_runtime(config, folder):
        runtime["config"] = config
        runtime["log_folder"] = folder
        runtime["enable_items"] = config["enable_items"]

        if agent:
            agent.log_folder = folder
            agent.qa_context_window = config["qa_context_window"]
            agent.enable_visual_context = config["enable_visual_context"]
            agent.set_game_name(config["game_name"])

        qa_handler = current_qa_handler(config)
        PASS = os.getenv("TWITCH_TOKEN")

        if not config["is_local_video"]:
            chat_listener = TwitchChatListener(
                config["twitch_username"],
                PASS,
                config["channel"],
                folder,
                start_time_ref=get_msg_time,
                question_handler=qa_handler,
                category_handler=update_stream_category,
                stream_start_handler=update_stream_start,
            )
            listener_thread = threading.Thread(target=chat_listener.listen, daemon=True)
            listener_thread.start()
            runtime["chat_listener"] = chat_listener
            runtime["listener_thread"] = listener_thread
            print(f"Connected to {config['channel']}. Listening for questions in the background...")
        else:
            runtime["chat_listener"] = None
            runtime["listener_thread"] = None
            print(f"Local video '{config['channel']}' detected. Twitch chat listener is disabled.")

        capture_stop = threading.Event()
        capture_thread = threading.Thread(
            target=my_streamlink.run_capture_loop,
            args=(config["source"], folder, config["process_fast"], qa_handler, capture_stop),
            daemon=True,
        )
        capture_thread.start()
        runtime["capture_stop"] = capture_stop
        runtime["capture_thread"] = capture_thread

    start_runtime(runtime_config, log_folder)

    
    while True:
        cmd = input("Cmd (status, clear, reload, quit)> ").strip().lower()
        
        if cmd in ("status", "list"):
            print("\n--- Status ---")

            print(f"Gemini Agent: {'ACTIVE' if agent else 'OFFLINE'}")
            print(f"Channel: {runtime['config']['channel']}")
            print(f"Capture: {'ACTIVE' if runtime['capture_thread'] and runtime['capture_thread'].is_alive() else 'OFFLINE'}")
            print(f"Chat Listener: {'ACTIVE' if runtime['chat_listener'] else 'OFFLINE'}")
            

            chat_listener = runtime["chat_listener"]
            if not chat_listener or not chat_listener.question_queue:
                print("Question queue is empty.")
            else:
                print("Queued questions:")
                for i, q in enumerate(chat_listener.question_queue):
                    print(f"[{i}] {q['user']}: {q['msg']}")
            print("--------------\n")
                
        elif cmd == "clear":
            if runtime["chat_listener"]:
                runtime["chat_listener"].question_queue.clear()
            print("Queue cleared.\n")

        elif cmd == "reload":
            try:
                new_config = resolve_config(reload_config())
            except Exception as e:
                print(f"Config reload failed: {e}\n")
                continue

            old_config = runtime["config"]
            changed_channel = new_config["source"] != old_config["source"]
            changed_capture = (
                changed_channel
                or new_config["process_fast"] != old_config["process_fast"]
                or new_config["enable_qa"] != old_config["enable_qa"]
            )

            if agent:
                agent.qa_context_window = new_config["qa_context_window"]
                agent.enable_visual_context = new_config["enable_visual_context"]
                agent.set_game_name(new_config["game_name"])
            runtime["enable_items"] = new_config["enable_items"]

            if changed_capture:
                uptime = time.time() - STREAM_START_TIME
                log_start_stop(runtime["log_folder"], "stop", uptime=uptime)
                stop_runtime()
                if changed_channel:
                    shared_deque.clear()
                    my_streamlink.video_frames.clear()

                stream_start = datetime.datetime.now()
                new_log_folder = create_stream_folder(new_config["stream_name"], stream_start)
                log_start_stop(new_log_folder, "start")

                STREAM_START_TIME = time.time()
                my_streamlink.STREAM_START_TIME = STREAM_START_TIME
                start_runtime(new_config, new_log_folder)
                print(f"Reloaded config and restarted stream workers for {new_config['channel']}.\n")
            else:
                runtime["config"] = new_config
                print("Reloaded config without restarting stream workers.\n")
            
        elif cmd == "quit":
            uptime = time.time() - STREAM_START_TIME
            log_start_stop(runtime["log_folder"], "stop", uptime=uptime)
            stop_runtime()
            print("Shutting down...")
            break

if __name__ == "__main__":
    main()
