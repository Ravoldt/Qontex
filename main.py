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
from utils import (
    TwitchCredentialsError,
    create_stream_folder,
    flush_json_buffer,
    load_config,
    log_start_stop,
    refresh_twitch_token,
    reload_config,
    shared_deque,
)
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
        print(f"[QONTEX] QA Agent is DISABLED ({reason}).")
        return

    if state.agent is not None and not provider_changed:
        if hasattr(state.agent, "refresh_from_state"):
            state.agent.refresh_from_state()
        return

    state.agent = create_qa_agent(state)
    if state.agent is None:
        print("[QONTEX] QA Agent is DISABLED via QA_AGENT.")
    else:
        print(f"[QONTEX] QA Agent loaded: {state.config.qa_agent}")


def main():
    parser = argparse.ArgumentParser(description="Qontex Stream AI Agent")
    parser.add_argument("--test-chat", action="store_true", help="Run only the Twitch Chat module")
    parser.add_argument("--test-capture", action="store_true", help="Run only the Video/Audio Capture module")
    parser.add_argument("--no-gemini", action="store_true", help="Run without sending context/questions to the QA agent")
    args = parser.parse_args()

    load_dotenv()

    try:
        config = load_config()
        runtime_config = apply_process_fast_safety(resolve_config(config))
        runtime_config, stream_start = hydrate_source_metadata(runtime_config)
    except FileNotFoundError:
        print("[QONTEX] CRITICAL ERROR: config.toml is missing!")
        exit(1)
    except ValueError as e:
        print(f"[QONTEX] {e}")
        exit(1)

    if runtime_config.is_livestream:
        try:
            refresh_twitch_token(strict=True)
        except TwitchCredentialsError as e:
            print(f"[QONTEX] Twitch credentials error: {e}")
            exit(1)

    if runtime_config.process_fast:
        print("[QONTEX] PROCESS_FAST is enabled; QA, item processing, and visual/audio context are disabled.")

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
        if not category:
            return
        if state.config.game_name != category:
            state.config = replace(state.config, game_name=category)
        if state.agent and hasattr(state.agent, "set_game_name"):
            state.agent.set_game_name(category)

    defer_live_models = runtime_config.is_livestream and not args.test_capture
    if defer_live_models:
        print("\n[QONTEX] Livestream mode: AI models will load when the stream is live.")
    else:
        print("\n[QONTEX] Preloading AI models into VRAM... (This will pause the script until ready)")

    if not args.test_chat and not defer_live_models:
        my_streamlink.preload_models(runtime_config.transcription_model)

    if not args.test_capture and not defer_live_models:
        from utils import preload_classifier

        if runtime_config.enable_question_detector:
            preload_classifier()

    if not defer_live_models:
        print("[QONTEX] All models successfully loaded!\n")

    if args.test_chat:
        if not runtime_config.is_livestream:
            print("[QONTEX] Cannot test Twitch chat with a VOD/offline source.")
            return
        print(f"[QONTEX] Running standalone Twitch Chat test for {runtime_config.channel}")
        listener = TwitchChatListener(
            runtime_config.twitch_username,
            os.getenv("TWITCH_TOKEN"),
            runtime_config.channel,
            log_folder,
            category_handler=lambda category: print(f"[QONTEX] Using Twitch category: {category}"),
            stream_start_handler=update_stream_start,
            state=state,
        )
        listener.listen()
        return

    if args.test_capture:
        print(f"[QONTEX] Running standalone Capture test for {runtime_config.source}")
        my_streamlink.run_capture_loop(
            runtime_config.source,
            log_folder,
            process_fast=runtime_config.process_fast,
            target_fps=runtime_config.visual_context_fps,
            state=state,
        )
        return

    if not runtime_config.is_livestream:
        try:
            configure_agent(state, no_gemini=args.no_gemini)
        except ValueError as e:
            print(f"[QONTEX] CRITICAL ERROR: {e}")
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
        print("[QONTEX] Item Processing is DISABLED via config.toml.")

    runtime_lock = threading.RLock()
    model_lock = threading.RLock()

    def current_qa_handler(config):
        return answer_question if config.enable_qa else None

    def set_chat_stream_active(active):
        chat_listener = state.chat_listener
        if chat_listener and hasattr(chat_listener, "set_stream_active"):
            chat_listener.set_stream_active(active)

    def load_runtime_models(config):
        with model_lock:
            if config.enable_qa or config.enable_items:
                if state.agent is None:
                    configure_agent(state, no_gemini=args.no_gemini)
                elif hasattr(state.agent, "refresh_from_state"):
                    state.agent.refresh_from_state()
            else:
                state.agent = None

            my_streamlink.preload_models(config.transcription_model)
            if config.enable_question_detector:
                from utils import preload_classifier

                preload_classifier()

    def unload_runtime_models(reason):
        with model_lock:
            print(f"[QONTEX] Stream is offline ({reason}). Unloading AI models from VRAM...")
            if state.agent is not None:
                for method_name in ("unload_models", "unload_model", "unload", "close"):
                    unload = getattr(state.agent, method_name, None)
                    if callable(unload):
                        try:
                            unload()
                        except Exception as e:
                            print(f"[QONTEX] [!] QA agent unload failed: {e}")
                        break
                state.agent = None

            my_streamlink.unload_models()
            from utils import unload_classifier

            unload_classifier()
            print("[QONTEX] AI models unloaded. Chat/EventSub listener remains active.")

    def stop_capture_worker(wait=True):
        capture_stop = state.capture_stop
        if capture_stop:
            capture_stop.set()

        capture_process = state.capture_process
        if capture_process and capture_process.poll() is None:
            try:
                capture_process.terminate()
                capture_process.wait(timeout=5)
            except Exception:
                try:
                    capture_process.kill()
                    capture_process.wait(timeout=5)
                except Exception:
                    pass

        capture_thread = state.capture_thread
        if wait and capture_thread and capture_thread.is_alive() and capture_thread is not threading.current_thread():
            capture_thread.join(timeout=10)

        if not capture_thread or not capture_thread.is_alive() or capture_thread is threading.current_thread():
            state.capture_thread = None
            state.capture_stop = None
            state.capture_process = None
            return True

        return False

    def handle_stream_offline(reason="eventsub"):
        if not state.config.is_livestream:
            return

        with runtime_lock:
            capture_alive = state.capture_thread and state.capture_thread.is_alive()
            if not state.stream_live and not capture_alive:
                set_chat_stream_active(False)
                return

            state.stream_live = False
            set_chat_stream_active(False)

        stopped = stop_capture_worker(wait=True)
        if not stopped:
            print("[QONTEX] [!] Capture thread did not stop cleanly; keeping models loaded to avoid interrupting active inference.")
            return
        flush_json_buffer(force=True)
        unload_runtime_models(reason)
        print(f"[QONTEX] Waiting for {state.config.channel} to come online again...")

    def start_capture_worker(config, folder, qa_handler_transcript, transcript_user):
        capture_stop = threading.Event()
        state.capture_stop = capture_stop

        def capture_runner():
            try:
                result = my_streamlink.run_capture_loop(
                    config.source,
                    folder,
                    process_fast=config.process_fast,
                    question_handler=qa_handler_transcript,
                    stop_event=capture_stop,
                    transcript_user=transcript_user,
                    target_fps=config.visual_context_fps,
                    state=state,
                )
                if result in ("stream_ended", "unavailable") and not capture_stop.is_set():
                    handle_stream_offline(result)
            finally:
                with runtime_lock:
                    if state.capture_thread is threading.current_thread():
                        state.capture_thread = None
                        state.capture_stop = None
                        state.capture_process = None

        capture_thread = threading.Thread(target=capture_runner, daemon=True)
        state.capture_thread = capture_thread
        capture_thread.start()

    def handle_stream_online(started_at=None):
        if not state.config.is_livestream:
            return

        with runtime_lock:
            if state.capture_thread and state.capture_thread.is_alive():
                state.stream_live = True
                set_chat_stream_active(True)
                return

            if started_at:
                update_stream_start(started_at)

            state.stream_live = True
            set_chat_stream_active(True)
            print(f"[QONTEX] {state.config.channel} is live. Loading models and starting capture...")

            try:
                load_runtime_models(state.config)
            except ValueError as e:
                print(f"[QONTEX] QA agent load failed: {e}")
                state.agent = None
            except Exception as e:
                print(f"[QONTEX] Failed to load runtime models: {e}")
                state.stream_live = False
                set_chat_stream_active(False)
                return

            qa_handler = current_qa_handler(state.config)
            qa_handler_transcript = qa_handler if state.config.enable_qa_transcript else None
            start_capture_worker(state.config, state.log_folder, qa_handler_transcript, state.config.stream_name)

    def stop_runtime():
        if state.chat_listener:
            state.chat_listener.stop()
            state.chat_listener = None

        state.stream_live = False
        stop_capture_worker(wait=True)

        if hasattr(state, "startup_thread") and state.startup_thread and state.startup_thread.is_alive():
            state.startup_thread.join(timeout=5)
            state.startup_thread = None

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
            state.stream_live = False
            chat_listener = TwitchChatListener(
                config.twitch_username,
                os.getenv("TWITCH_TOKEN"),
                config.channel,
                folder,
                question_handler=qa_handler_chat,
                category_handler=update_stream_category,
                stream_start_handler=update_stream_start,
                stream_online_handler=handle_stream_online,
                stream_offline_handler=handle_stream_offline,
                state=state,
            )
            chat_listener.set_stream_active(False)
            listener_thread = threading.Thread(target=chat_listener.listen, daemon=True)
            listener_thread.start()
            state.chat_listener = chat_listener
            state.listener_thread = listener_thread
            state.startup_thread = None
            print(f"[QONTEX] Connected to {config.channel} chat/EventSub. Waiting for stream status...")
            return

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
            print(f"[QONTEX] VOD source '{config.source}' detected. Using chat file: {chat_path}")
        else:
            state.chat_listener = None
            state.listener_thread = None
            print(f"[QONTEX] VOD source '{config.source}' detected. Offline chat is disabled. (No chat file found)")

        start_capture_worker(config, folder, qa_handler_transcript, transcript_user)

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
                            print("[QONTEX] QA Agent is DISABLED while PROCESS_FAST is enabled.")
                        elif state.agent:
                            print(f"[QONTEX] Asking QA agent: {question}")
                            direct_ask = getattr(state.agent, "direct_ask", None)
                            if not callable(direct_ask):
                                print("[QONTEX] Configured QA agent does not implement direct_ask.")
                                continue
                            threading.Thread(
                                target=direct_ask,
                                args=(question,),
                                kwargs={"timestamp": state.current_timestamp()},
                                daemon=True,
                            ).start()
                        else:
                            print("[QONTEX] QA Agent is DISABLED.")
                    else:
                        print("[QONTEX] Usage: /ask <question>")
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
                        print(f"[QONTEX] Updated {key} to {val_str} in {target_path}")
                        cmd = "reload"
                    except Exception as e:
                        print(f"[QONTEX] Failed to update config.toml: {e}")
                        continue
                else:
                    print(f"[QONTEX] Usage: /{command} <value>")
                    continue

            if cmd in ("status", "list"):
                print("\n--- Status ---")

                print(f"QA Agent: {'ACTIVE' if state.agent else 'OFFLINE'}")
                print(f"Source: {state.config.source_kind} {state.config.source}")
                print(f"Chat File: {state.config.chat_path or 'none'}")
                capture_active = state.capture_thread and state.capture_thread.is_alive()
                if state.config.is_livestream and state.chat_listener and not capture_active and not state.stream_live:
                    print("Capture: WAITING FOR STREAM")
                    print("Chat Listener: ACTIVE")
                    print("Models: UNLOADED")
                else:
                    print(f"Capture: {'ACTIVE' if capture_active else 'OFFLINE'}")
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
                    print(f"[QONTEX] Config reload failed: {e}\n")
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
                    or new_config.transcription_model != old_config.transcription_model
                    or new_config.mega_asr_path != old_config.mega_asr_path
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
                        print(f"[QONTEX] QA agent reload failed: {e}")
                        state.agent = None

                    start_runtime(new_config, new_log_folder)
                    print(f"[QONTEX] Reloaded config and restarted stream workers for {new_config.source}.\n")
                else:
                    state.refresh_config(new_config)
                    try:
                        configure_agent(state, no_gemini=args.no_gemini, previous_config=old_config)
                    except ValueError as e:
                        print(f"[QONTEX] QA agent reload failed: {e}")
                        state.agent = None
                    print("[QONTEX] Reloaded config without restarting stream workers.\n")

            elif cmd == "quit":
                uptime = time.time() - state.clock_start_wall
                log_start_stop(state.log_folder, "stop", uptime=uptime)
                stop_runtime()
                print("[QONTEX] Shutting down...")
                break

    except KeyboardInterrupt:
        print("\n[QONTEX] Keyboard interrupt received. Shutting down...")
        uptime = time.time() - state.clock_start_wall
        log_start_stop(state.log_folder, "stop", uptime=uptime)
        stop_runtime()


if __name__ == "__main__":
    main()
