import argparse
import datetime
import json
import os
import re
import threading
import time
from dataclasses import replace

from dotenv import load_dotenv

from qa_agent import create_qa_agent
from stream_state import StreamConfig, StreamState
from twitch_chat import TwitchChatListener
from utils import create_stream_folder, flush_json_buffer, load_config, log_start_stop, refresh_twitch_token, reload_config, shared_deque
import streamcapture as my_streamlink


def resolve_config(config):
    return StreamConfig.from_raw(config)


def apply_process_fast_safety(config):
    if not config.process_fast:
        return config

    return replace(
        config,
        enable_qa=False,
        enable_qa_chat=False,
        enable_qa_transcript=False,
        enable_items=False,
        enable_visual_context=False,
        enable_audio_context=False,
    )


def hydrate_source_metadata(config):
    stream_start = datetime.datetime.now()
    if not config.is_vod or not config.chat_path or not os.path.exists(config.chat_path):
        return config, stream_start

    from twitch_chat import LocalChatListener

    temp_listener = LocalChatListener(
        config.chat_path,
        None,
        process_fast=config.process_fast,
        initialize_sync=False,
    )
    login_name = temp_listener.streamer_login or temp_listener.streamer_name
    if login_name:
        config = replace(config, stream_name=login_name)
    if temp_listener.stream_start_date:
        stream_start = temp_listener.stream_start_date
    return config, stream_start


def configure_agent(state, no_gemini=False, previous_config=None):
    disabled = no_gemini or state.config.process_fast
    provider_changed = previous_config is None or state.config.qa_agent != previous_config.qa_agent

    if disabled:
        state.agent = None
        reason = "PROCESS_FAST is enabled" if state.config.process_fast else "--no-gemini flag used"
        print(f"QA Agent is DISABLED ({reason}).")
        return

    if state.agent is not None and not provider_changed:
        if hasattr(state.agent, "refresh_from_state"):
            state.agent.refresh_from_state()
        return

    state.agent = create_qa_agent(state)
    if state.agent is None:
        print("QA Agent is DISABLED via QA_AGENT.")
    else:
        print(f"QA Agent loaded: {state.config.qa_agent}")


def main():
    parser = argparse.ArgumentParser(description="Qontex Stream AI Agent")
    parser.add_argument("--test-chat", action="store_true", help="Run only the Twitch Chat module")
    parser.add_argument("--test-capture", action="store_true", help="Run only the Video/Audio Capture module")
    parser.add_argument("--no-gemini", action="store_true", help="Run without sending context/questions to the QA agent")
    args = parser.parse_args()

    load_dotenv()
    refresh_twitch_token()

    try:
        config = load_config()
        runtime_config = apply_process_fast_safety(resolve_config(config))
        runtime_config, stream_start = hydrate_source_metadata(runtime_config)
    except FileNotFoundError:
        print("CRITICAL ERROR: config.toml is missing!")
        exit(1)
    except ValueError as e:
        print(e)
        exit(1)

    if runtime_config.process_fast:
        print("PROCESS_FAST is enabled; QA, item processing, and visual/audio context are disabled.")

    log_folder = create_stream_folder(runtime_config.stream_name, stream_start)
    log_start_stop(log_folder, "start")

    state = StreamState(
        config=runtime_config,
        log_folder=log_folder,
        stream_start=stream_start,
        timeline=shared_deque,
    )
    my_streamlink.STREAM_START_TIME = state.clock_start_wall

    def update_stream_start(ts):
        state.update_stream_start(ts)
        my_streamlink.STREAM_START_TIME = state.clock_start_wall

    def answer_question(msg):
        if state.agent and hasattr(state.agent, "answer_question"):
            state.agent.answer_question(msg)

    def update_stream_category(category):
        if state.agent and hasattr(state.agent, "set_game_name"):
            state.agent.set_game_name(category)

    print("\nPreloading AI models into VRAM... (This will pause the script until ready)")
    if not args.test_chat:
        my_streamlink.preload_models()

    if not args.test_capture:
        from utils import preload_classifier

        if runtime_config.enable_question_detector:
            preload_classifier()
    print("All models successfully loaded!\n")

    if args.test_chat:
        if not runtime_config.is_livestream:
            print("Cannot test Twitch chat with a VOD/offline source.")
            return
        print(f"Running standalone Twitch Chat test for {runtime_config.channel}")
        listener = TwitchChatListener(
            runtime_config.twitch_username,
            os.getenv("TWITCH_TOKEN"),
            runtime_config.channel,
            log_folder,
            category_handler=lambda category: print(f"Using Twitch category: {category}"),
            stream_start_handler=update_stream_start,
            state=state,
        )
        listener.listen()
        return

    if args.test_capture:
        print(f"Running standalone Capture test for {runtime_config.source}")
        my_streamlink.run_capture_loop(
            runtime_config.source,
            log_folder,
            process_fast=runtime_config.process_fast,
            target_fps=runtime_config.visual_context_fps,
            state=state,
        )
        return

    try:
        configure_agent(state, no_gemini=args.no_gemini)
    except ValueError as e:
        print(f"CRITICAL ERROR: {e}")
        exit(1)

    def item_processor():
        while True:
            time.sleep(30)
            process_items = getattr(state.agent, "process_items", None)
            if state.enable_items and callable(process_items):
                process_items()

    processor_thread = threading.Thread(target=item_processor, daemon=True)
    processor_thread.start()
    if not runtime_config.enable_items:
        print("Item Processing is DISABLED via config.toml.")

    def current_qa_handler(config):
        return answer_question if (state.agent and config.enable_qa) else None

    def stop_runtime():
        if state.chat_listener:
            state.chat_listener.stop()
            state.chat_listener = None

        if state.capture_stop:
            state.capture_stop.set()

        if state.capture_thread and state.capture_thread.is_alive():
            state.capture_thread.join(timeout=5)

        state.capture_thread = None
        state.capture_stop = None

        flush_json_buffer(force=True)

    def start_runtime(config, folder):
        state.refresh_config(config, folder)
        if state.agent and hasattr(state.agent, "refresh_from_state"):
            state.agent.refresh_from_state()

        qa_handler = current_qa_handler(config)
        qa_handler_chat = qa_handler if config.enable_qa_chat else None
        qa_handler_transcript = qa_handler if config.enable_qa_transcript else None

        transcript_user = config.stream_name

        if config.is_livestream:
            chat_listener = TwitchChatListener(
                config.twitch_username,
                os.getenv("TWITCH_TOKEN"),
                config.channel,
                folder,
                question_handler=qa_handler_chat,
                category_handler=update_stream_category,
                stream_start_handler=update_stream_start,
                state=state,
            )
            listener_thread = threading.Thread(target=chat_listener.listen, daemon=True)
            listener_thread.start()
            state.chat_listener = chat_listener
            state.listener_thread = listener_thread
            print(f"Connected to {config.channel}. Listening for questions in the background...")
        else:
            chat_path = config.chat_path
            if chat_path and os.path.exists(chat_path):
                from twitch_chat import LocalChatListener

                chat_listener = LocalChatListener(
                    chat_path,
                    folder,
                    question_handler=qa_handler_chat,
                    process_fast=config.process_fast,
                    state=state,
                )
                login_name = chat_listener.streamer_login or chat_listener.streamer_name
                if login_name:
                    transcript_user = login_name
                    state.config = replace(state.config, stream_name=login_name)
                    if state.agent and hasattr(state.agent, "streamer_name"):
                        state.agent.streamer_name = login_name

                listener_thread = threading.Thread(target=chat_listener.listen, daemon=True)
                listener_thread.start()
                state.chat_listener = chat_listener
                state.listener_thread = listener_thread
                print(f"VOD source '{config.source}' detected. Using chat file: {chat_path}")
            else:
                state.chat_listener = None
                state.listener_thread = None
                print(f"VOD source '{config.source}' detected. Offline chat is disabled. (No chat file found)")

        capture_stop = threading.Event()
        capture_thread = threading.Thread(
            target=my_streamlink.run_capture_loop,
            args=(config.source, folder),
            kwargs={
                "process_fast": config.process_fast,
                "question_handler": qa_handler_transcript,
                "stop_event": capture_stop,
                "transcript_user": transcript_user,
                "target_fps": config.visual_context_fps,
                "state": state,
            },
            daemon=True,
        )
        capture_thread.start()
        state.capture_stop = capture_stop
        state.capture_thread = capture_thread

    start_runtime(runtime_config, log_folder)

    try:
        while True:
            raw_cmd = input("Cmd (status, timeline, clear, reload, quit, /<config> <val>, /ask <q>)> ").strip()
            if not raw_cmd:
                continue

            cmd = raw_cmd.lower()

            if raw_cmd.startswith("/"):
                parts = raw_cmd[1:].split(maxsplit=1)
                if not parts:
                    continue
                command = parts[0].lower()

                if command == "ask":
                    if len(parts) > 1:
                        question = parts[1]
                        if state.config.process_fast:
                            print("QA Agent is DISABLED while PROCESS_FAST is enabled.")
                        elif state.agent:
                            print(f"Asking QA agent: {question}")
                            direct_ask = getattr(state.agent, "direct_ask", None)
                            if not callable(direct_ask):
                                print("Configured QA agent does not implement direct_ask.")
                                continue
                            threading.Thread(
                                target=direct_ask,
                                args=(question,),
                                kwargs={"timestamp": state.current_timestamp()},
                                daemon=True,
                            ).start()
                        else:
                            print("QA Agent is DISABLED.")
                    else:
                        print("Usage: /ask <question>")
                    continue

                if len(parts) > 1:
                    key = command.upper()
                    val_str = parts[1].strip()

                    if val_str.lower() == "true":
                        val_str = "true"
                    elif val_str.lower() == "false":
                        val_str = "false"
                    elif val_str.isdigit():
                        pass
                    elif not (val_str.startswith('"') and val_str.endswith('"')):
                        val_str = f'"{val_str}"'

                    try:
                        target_path = "config.toml"
                        local_path = "local.toml"

                        if os.path.exists(local_path):
                            with open(local_path, "r", encoding="utf-8") as f:
                                if re.search(rf"(?mi)^[ \t]*{re.escape(key)}[ \t]*=", f.read()):
                                    target_path = local_path

                        with open(target_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        pattern = re.compile(rf"(?mi)^[ \t]*{re.escape(key)}[ \t]*=.*$")
                        if pattern.search(content):
                            content = pattern.sub(f"{key} = {val_str}", content)
                        else:
                            content = content.rstrip() + f"\n{key} = {val_str}\n"

                        with open(target_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        print(f"Updated {key} to {val_str} in {target_path}")
                        cmd = "reload"
                    except Exception as e:
                        print(f"Failed to update config.toml: {e}")
                        continue
                else:
                    print(f"Usage: /{command} <value>")
                    continue

            if cmd in ("status", "list"):
                print("\n--- Status ---")

                print(f"QA Agent: {'ACTIVE' if state.agent else 'OFFLINE'}")
                print(f"Source: {state.config.source_kind} {state.config.source}")
                print(f"Chat File: {state.config.chat_path or 'none'}")
                print(f"Capture: {'ACTIVE' if state.capture_thread and state.capture_thread.is_alive() else 'OFFLINE'}")
                print(f"Chat Listener: {'ACTIVE' if state.chat_listener else 'OFFLINE'}")

                chat_listener = state.chat_listener
                if not chat_listener or not chat_listener.question_queue:
                    print("Question queue is empty.")
                else:
                    print("Queued questions:")
                    for i, q in enumerate(chat_listener.question_queue):
                        print(f"[{i}] {q['user']}: {q['msg']}")
                print("--------------\n")

            elif cmd == "timeline":
                log_path = os.path.join(state.log_folder, "merged.json")
                flush_json_buffer(force=True)
                if os.path.exists(log_path):
                    try:
                        lines = []
                        with open(log_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    try:
                                        lines.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        continue

                        lines.sort(key=lambda x: x.get("timestamp", 0))

                        print("\n--- Chronological Timeline ---")
                        for entry in lines:
                            ts = max(0, int(entry.get("timestamp", 0)))
                            hours, remainder = divmod(ts, 3600)
                            minutes, seconds = divmod(remainder, 60)
                            ts_str = f"[{hours}:{minutes:02d}:{seconds:02d}]"
                            user = entry.get("user") or entry.get("type", "UNKNOWN").upper()
                            print(f"{ts_str} {user}: {entry.get('text', '')}")
                        print("------------------------------\n")
                    except Exception as e:
                        print(f"Error reading timeline: {e}\n")
                else:
                    print("No merged.json found for the current session yet.\n")

            elif cmd == "clear":
                if state.chat_listener:
                    state.chat_listener.question_queue.clear()
                print("Queue cleared.\n")

            elif cmd == "reload":
                try:
                    new_config = apply_process_fast_safety(resolve_config(reload_config()))
                    new_config, new_stream_start = hydrate_source_metadata(new_config)
                except Exception as e:
                    print(f"Config reload failed: {e}\n")
                    continue

                old_config = state.config
                changed_source = (
                    new_config.source != old_config.source
                    or new_config.source_kind != old_config.source_kind
                    or new_config.chat_path != old_config.chat_path
                )
                changed_capture = (
                    changed_source
                    or new_config.process_fast != old_config.process_fast
                    or new_config.enable_qa != old_config.enable_qa
                    or new_config.enable_qa_chat != old_config.enable_qa_chat
                    or new_config.enable_qa_transcript != old_config.enable_qa_transcript
                    or new_config.enable_question_detector != old_config.enable_question_detector
                    or new_config.question_detector_type != old_config.question_detector_type
                    or new_config.enable_visual_context != old_config.enable_visual_context
                    or new_config.visual_context_fps != old_config.visual_context_fps
                )

                if changed_capture:
                    uptime = time.time() - state.clock_start_wall
                    log_start_stop(state.log_folder, "stop", uptime=uptime)
                    stop_runtime()
                    if changed_source:
                        state.clear_buffers()
                        my_streamlink.video_frames.clear()
                        my_streamlink.audio_chunks.clear()

                    new_log_folder = create_stream_folder(new_config.stream_name, new_stream_start)
                    log_start_stop(new_log_folder, "start")
                    state.refresh_config(new_config, new_log_folder)
                    state.stream_start = new_stream_start
                    state.clock_start_wall = time.time()
                    my_streamlink.STREAM_START_TIME = state.clock_start_wall

                    try:
                        configure_agent(state, no_gemini=args.no_gemini, previous_config=old_config)
                    except ValueError as e:
                        print(f"QA agent reload failed: {e}")
                        state.agent = None

                    start_runtime(new_config, new_log_folder)
                    print(f"Reloaded config and restarted stream workers for {new_config.source}.\n")
                else:
                    state.refresh_config(new_config)
                    try:
                        configure_agent(state, no_gemini=args.no_gemini, previous_config=old_config)
                    except ValueError as e:
                        print(f"QA agent reload failed: {e}")
                        state.agent = None
                    print("Reloaded config without restarting stream workers.\n")

            elif cmd == "quit":
                uptime = time.time() - state.clock_start_wall
                log_start_stop(state.log_folder, "stop", uptime=uptime)
                stop_runtime()
                print("Shutting down...")
                break

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down...")
        uptime = time.time() - state.clock_start_wall
        log_start_stop(state.log_folder, "stop", uptime=uptime)
        stop_runtime()


if __name__ == "__main__":
    main()
