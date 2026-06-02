import asyncio
import datetime
import json
import os
import time
import threading
import twitchio
from twitchio.ext import commands

from utils import (
    Message,
    TwitchCredentialsError,
    TWITCH_REQUIRED_SCOPES,
    create_stream_folder,
    format_twitch_scopes,
    get_config_value,
    is_likely_question,
    load_config,
    log_json,
    log_message,
    normalize_twitch_token,
    shared_deque,
    refresh_twitch_token,
    twitch_reauthorization_hint,
)


class TwitchChatListener(commands.Bot):
    def __init__(
        self,
        nick,
        password,
        channel,
        log_folder,
        start_time_ref=None,
        question_handler=None,
        category_handler=None,
        stream_start_handler=None,
        stream_online_handler=None,
        stream_offline_handler=None,
        state=None,
    ):
        self.state = state
        self.bot_nick = nick
        self.channel = channel.lstrip("#").lower()
        self.log_folder = log_folder or (state.log_folder if state else None)
        self.start_time_ref = start_time_ref or (state.current_timestamp if state else None)
        self.question_handler = question_handler
        self.category_handler = category_handler
        self.stream_start_handler = stream_start_handler
        self.stream_online_handler = stream_online_handler
        self.stream_offline_handler = stream_offline_handler
        self.question_queue = []
        self.stream_category = None
        self.stream_active = True
        self.broadcaster_id = None
        self._last_live = None
        self._fallback_start = time.time()
        self._info_loop_started = False
        self._info_task = None
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)

        self.twitch_token = self._normalize_token(password)
        super().__init__(
            client_id=os.getenv("TWITCH_CLIENT_ID"),
            client_secret=os.getenv("TWITCH_CLIENT_SECRET"),
            bot_id=os.getenv("TWITCH_BOT_ID"),
            prefix="!",
        )

    def _normalize_token(self, token):
        token = normalize_twitch_token(token)
        if not token:
            raise TwitchCredentialsError(f"TWITCH_TOKEN is required for Twitch chat. {twitch_reauthorization_hint()}")
        return token

    async def setup_hook(self):
        refresh_token = os.getenv("TWITCH_REFRESH_TOKEN")
        if not refresh_token:
            raise TwitchCredentialsError(
                f"TWITCH_REFRESH_TOKEN is required for Twitch EventSub chat. {twitch_reauthorization_hint()}"
            )

        validated = await self.add_token(self.twitch_token, refresh_token)
        client_id = os.getenv("TWITCH_CLIENT_ID")
        if validated.client_id != client_id:
            raise TwitchCredentialsError(
                "TWITCH_TOKEN was issued for client ID "
                f"{validated.client_id}, but TWITCH_CLIENT_ID is {client_id}."
            )

        if validated.user_id != self.bot_id:
            raise TwitchCredentialsError(
                "TWITCH_TOKEN belongs to user ID "
                f"{validated.user_id}, but TWITCH_BOT_ID is {self.bot_id}."
            )

        missing_scopes = TWITCH_REQUIRED_SCOPES - set(validated.scopes)
        if missing_scopes:
            raise TwitchCredentialsError(
                f"TWITCH_TOKEN must include these scopes: {format_twitch_scopes(missing_scopes)}. "
                f"{twitch_reauthorization_hint()}"
            )

    def is_likely_question(self, message, msg_type="chat"):
        return is_likely_question(message, msg_type)

    def handle_question(self, msg):
        self.question_queue.append({"user": msg.user, "msg": msg.text, "timestamp": msg.timestamp})
        if len(self.question_queue) > 20:
            self.question_queue.pop(0)
        if self.question_handler:
            threading.Thread(target=self.question_handler, args=(msg,), daemon=True).start()

    def set_stream_active(self, active):
        self.stream_active = bool(active)

    def listen(self):
        asyncio.set_event_loop(self._event_loop)
        try:
            self._event_loop.run_until_complete(self.start())
        except TwitchCredentialsError as e:
            print(f"[QONTEX] Twitch chat/EventSub startup failed: {e}")

    def stop(self):
        if self._event_loop.is_running():
            if getattr(self, '_info_task', None) and not self._info_task.done():
                self._event_loop.call_soon_threadsafe(self._info_task.cancel)
            self._event_loop.call_soon_threadsafe(lambda: asyncio.create_task(self.close()))

    async def event_ready(self):
        print(f"[QONTEX] Logged in to Twitch as | {self.bot_nick}")

        # 1. Fetch the streamer's user info to get their numeric Broadcaster ID
        users = await self.fetch_users(logins=[self.channel])
        if not users:
            print(f"[QONTEX] Could not find Twitch channel: {self.channel}")
            return
        broadcaster_id = users[0].id
        self.broadcaster_id = broadcaster_id

        subscription = twitchio.eventsub.ChatMessageSubscription(
            broadcaster_user_id=broadcaster_id,
            user_id=self.bot_id
        )

        await self.subscribe_websocket(payload=subscription)
        print(f"[QONTEX] Successfully subscribed to live chat for {self.channel}!")

        await self.subscribe_websocket(
            payload=twitchio.eventsub.StreamOnlineSubscription(broadcaster_user_id=broadcaster_id),
            as_bot=True,
        )
        await self.subscribe_websocket(
            payload=twitchio.eventsub.StreamOfflineSubscription(broadcaster_user_id=broadcaster_id),
            as_bot=True,
        )

        if not self._info_loop_started:
            self._info_loop_started = True
            self._info_task = asyncio.create_task(self.refresh_stream_info_loop())

    async def event_message(self, message):  # Changed parameter to 'message' for consistency with twitchio
        author = getattr(message, "author", None) or getattr(message, "chatter", None)
        username = getattr(author, "name", None) or "unknown"
        chat_msg = (getattr(message, "content", None) or getattr(message, "text", "")).strip()
        if not chat_msg:
            return

        msg_time = self.start_time_ref() if self.start_time_ref else (time.time() - self._fallback_start)
        msg = Message(msg_time, "chat", chat_msg, user=username)

        print(f"\r{msg}")

        log_message(self.log_folder, "chat.log", msg)
        log_json(self.log_folder, "merged.json", msg.to_dict())
        if self.state:
            self.state.add_message(msg)
        else:
            shared_deque.add_message(msg)

        should_detect_question = self.stream_active and (
            self.question_handler or get_config_value("LOG_QUESTION_DETECTIONS", True)
        )
        if should_detect_question and self.is_likely_question(msg.text, msg.type):
            self.handle_question(msg)

    async def event_stream_online(self, payload):
        self.set_stream_active(True)
        self._last_live = True
        print(f"\n[QONTEX] [*] Stream online event received for {payload.broadcaster.name}.")
        if self.stream_start_handler:
            self.stream_start_handler(payload.started_at)
        if self.stream_online_handler:
            threading.Thread(target=self.stream_online_handler, args=(payload.started_at,), daemon=True).start()

    async def event_stream_offline(self, payload):
        self.set_stream_active(False)
        self._last_live = False
        print(f"\n[QONTEX] [*] Stream offline event received for {payload.broadcaster.name}.")
        if self.stream_offline_handler:
            threading.Thread(target=self.stream_offline_handler, args=("eventsub",), daemon=True).start()

    async def refresh_stream_info_loop(self):
        try:
            while True:
                await self.refresh_stream_info()
                await asyncio.sleep(300)
        except asyncio.CancelledError:
            pass

    async def refresh_stream_info(self):
        category, started_at = await self.fetch_stream_info()
        if category and category != self.stream_category:
            self.stream_category = category
            if self.category_handler:
                self.category_handler(category)
            print(f"\rTwitch category: {category}")

        is_live = started_at is not None
        initial_state = self._last_live is None
        changed_state = not initial_state and self._last_live != is_live
        self._last_live = is_live
        self.set_stream_active(is_live)

        if is_live:
            if self.stream_start_handler:
                self.stream_start_handler(started_at)
            if (initial_state or changed_state) and self.stream_online_handler:
                threading.Thread(target=self.stream_online_handler, args=(started_at,), daemon=True).start()
        elif changed_state and self.stream_offline_handler:
            threading.Thread(target=self.stream_offline_handler, args=("poll",), daemon=True).start()
        return category, started_at

    async def fetch_stream_info(self):
        try:
            result = self.fetch_streams(user_logins=[self.channel])
            if hasattr(result, "__aiter__"):
                async for stream in result:
                    return getattr(stream, "game_name", None), getattr(stream, "started_at", None)
            else:
                streams = await result
                if streams:
                    return getattr(streams[0], "game_name", None), getattr(streams[0], "started_at", None)

            broadcaster_id = await self.resolve_broadcaster_id()
            if not broadcaster_id:
                return None, None

            channel_info = await self.fetch_channel(broadcaster_id)
            return getattr(channel_info, "game_name", None), None
        except Exception as e:
            print(f"\r[QONTEX] [!] Twitch stream info lookup failed: {e}")
        return None, None

    async def resolve_broadcaster_id(self):
        if self.broadcaster_id:
            return self.broadcaster_id

        users = await self.fetch_users(logins=[self.channel])
        if not users:
            print(f"\r[QONTEX] [!] Could not find Twitch channel: {self.channel}")
            return None

        self.broadcaster_id = users[0].id
        return self.broadcaster_id


class LocalChatListener:
    def __init__(self, json_path, log_folder, question_handler=None, process_fast=False, state=None, initialize_sync=True):
        self.state = state
        self.json_path = json_path
        self.log_folder = log_folder or (state.log_folder if state else None)
        self.question_handler = question_handler
        self.process_fast = process_fast
        self.messages = []
        self._stop_event = threading.Event()
        self.question_queue = []
        self.streamer_name = None
        self.streamer_login = None
        self.stream_start_date = None
        self._load_messages()

        if initialize_sync:
            if self.messages:
                self._set_chat_processing_timestamp(self.messages[0]["timestamp"])
            else:
                self._set_chat_processing_timestamp(float('inf'))

    def _load_messages(self):
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return

                # Check if it's JSON lines
                if content.startswith('{') and '\n' in content:
                    lines = content.split('\n')
                    for line in lines:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            if not self.stream_start_date and "created_at" in data:
                                try:
                                    dt_str = data["created_at"].replace("Z", "").split(".")[0]
                                    self.stream_start_date = datetime.datetime.fromisoformat(dt_str)
                                except Exception:
                                    pass
                            ts = data.get("timestamp", data.get("content_offset_seconds", 0))
                            user = data.get("user", data.get("commenter", {}).get("name", "unknown"))
                            text = data.get("text", data.get("message", {}).get("body", ""))
                            if "type" in data and data.get("type") != "chat":
                                continue
                            if text:
                                self.messages.append({"timestamp": float(ts), "user": user, "text": text})
                        except:
                            pass
                else:
                    # Try full JSON
                    data = json.loads(content)
                    if isinstance(data, dict):
                        if "streamer" in data and isinstance(data["streamer"], dict):
                            self.streamer_name = data["streamer"].get("name")
                            self.streamer_login = data["streamer"].get("login")
                        elif "video" in data and isinstance(data["video"], dict):
                            self.streamer_name = data["video"].get("user_name")
                            self.streamer_login = data["video"].get("user_login")

                        if "comments" in data:
                            # TwitchDownloader format
                            for comment in data["comments"]:
                                if not self.stream_start_date and "created_at" in comment:
                                    try:
                                        dt_str = comment["created_at"].replace("Z", "").split(".")[0]
                                        self.stream_start_date = datetime.datetime.fromisoformat(dt_str)
                                    except Exception:
                                        pass
                                ts = comment.get("content_offset_seconds", 0)
                                user = comment.get("commenter", {}).get("name", "unknown")
                                text = comment.get("message", {}).get("body", "")
                                if text:
                                    self.messages.append({"timestamp": float(ts), "user": user, "text": text})
                    elif isinstance(data, list):
                        for item in data:
                            if not self.stream_start_date and "created_at" in item:
                                try:
                                    dt_str = item["created_at"].replace("Z", "").split(".")[0]
                                    self.stream_start_date = datetime.datetime.fromisoformat(dt_str)
                                except Exception:
                                    pass
                            ts = item.get("timestamp", item.get("content_offset_seconds", 0))
                            user = item.get("user", item.get("commenter", {}).get("name", "unknown"))
                            text = item.get("text", item.get("message", {}).get("body", ""))
                            if "type" in item and item.get("type") != "chat":
                                continue
                            if text:
                                self.messages.append({"timestamp": float(ts), "user": user, "text": text})

            self.messages.sort(key=lambda x: x["timestamp"])
        except Exception as e:
            print(f"[QONTEX] Error loading local chat JSON: {e}")

    def listen(self):
        print(f"[QONTEX] Loaded {len(self.messages)} chat messages from {self.json_path}")
        msg_idx = 0

        try:
            while not self._stop_event.is_set() and msg_idx < len(self.messages):
                target_ts = self.messages[msg_idx]["timestamp"]
                self._set_chat_processing_timestamp(target_ts)

                current_ts = self._current_video_timestamp()

                if current_ts < target_ts:
                    time.sleep(0.01)
                    continue
                self._process_message(self.messages[msg_idx])
                msg_idx += 1

                if msg_idx < len(self.messages):
                    self._set_chat_processing_timestamp(self.messages[msg_idx]["timestamp"])
                else:
                    self._set_chat_processing_timestamp(float('inf'))
        finally:
            self._set_chat_processing_timestamp(float('inf'))

    def _process_message(self, msg_data):
        msg = Message(msg_data["timestamp"], "chat", msg_data["text"], user=msg_data["user"])
        print(f"\r{msg}")
        log_message(self.log_folder, "chat.log", msg)
        log_json(self.log_folder, "merged.json", msg.to_dict())
        if self.state:
            self.state.add_message(msg)
        else:
            shared_deque.add_message(msg)

        should_detect_question = self.question_handler or get_config_value("LOG_QUESTION_DETECTIONS", True)
        if should_detect_question and is_likely_question(msg.text, msg.type):
            self.question_queue.append({"user": msg.user, "msg": msg.text, "timestamp": msg.timestamp})
            if len(self.question_queue) > 20:
                self.question_queue.pop(0)
            if self.question_handler:
                threading.Thread(target=self.question_handler, args=(msg,), daemon=True).start()

    def stop(self):
        self._stop_event.set()

    def _set_chat_processing_timestamp(self, timestamp):
        if self.state:
            self.state.chat_processing_timestamp = timestamp
            return
        import streamcapture

        streamcapture.chat_processing_timestamp = timestamp

    def _current_video_timestamp(self):
        if self.state:
            return self.state.current_video_timestamp
        import streamcapture

        return streamcapture.current_video_timestamp

class StreamOnlineWaiter(twitchio.Client):
    def __init__(self, channel):
        self.channel = channel.lstrip("#").lower()
        self.twitch_token = self._normalize_token(os.getenv("TWITCH_TOKEN"))
        self.is_live = False
        self.waiter_stop = asyncio.Event()
        super().__init__(
            client_id=os.getenv("TWITCH_CLIENT_ID"),
            client_secret=os.getenv("TWITCH_CLIENT_SECRET"),
            bot_id=os.getenv("TWITCH_BOT_ID"),
        )

    def _normalize_token(self, token):
        return normalize_twitch_token(token)

    async def setup_hook(self):
        refresh_token = os.getenv("TWITCH_REFRESH_TOKEN")
        if not refresh_token:
            raise TwitchCredentialsError(f"TWITCH_REFRESH_TOKEN is required for EventSub. {twitch_reauthorization_hint()}")
        validated = await self.add_token(self.twitch_token, refresh_token)
        client_id = os.getenv("TWITCH_CLIENT_ID")
        if validated.client_id != client_id:
            raise TwitchCredentialsError(
                "TWITCH_TOKEN was issued for client ID "
                f"{validated.client_id}, but TWITCH_CLIENT_ID is {client_id}."
            )

        bot_id = os.getenv("TWITCH_BOT_ID")
        if bot_id and validated.user_id != bot_id:
            raise TwitchCredentialsError(
                "TWITCH_TOKEN belongs to user ID "
                f"{validated.user_id}, but TWITCH_BOT_ID is {bot_id}."
            )

        missing_scopes = TWITCH_REQUIRED_SCOPES - set(validated.scopes)
        if missing_scopes:
            raise TwitchCredentialsError(
                f"TWITCH_TOKEN must include these scopes: {format_twitch_scopes(missing_scopes)}. "
                f"{twitch_reauthorization_hint()}"
            )

    async def event_ready(self):
        users = await self.fetch_users(logins=[self.channel])
        if not users:
            print(f"\n[QONTEX] [!] Could not find channel: {self.channel}")
            self.waiter_stop.set()
            return

        broadcaster_id = users[0].id
        streams = await self.fetch_streams(user_ids=[broadcaster_id])
        if streams:
            self.is_live = True
            self.waiter_stop.set()
            return

        print(f"\n[QONTEX] [*] {self.channel} is currently offline. Waiting for stream to go live...")
        try:
            sub = twitchio.eventsub.StreamOnlineSubscription(broadcaster_user_id=broadcaster_id)
            await self.subscribe_websocket(payload=sub, as_bot=True)
        except Exception as e:
            print(f"[QONTEX] [!] EventSub subscription failed: {e}")
            self.waiter_stop.set()

    async def event_eventsub_notification_stream_start(self, payload):
        print(f"\n[QONTEX] [*] Stream online event received for {payload.broadcaster.name}! Starting main script...")
        self.is_live = True
        self.waiter_stop.set()

    async def event_stream_online(self, payload):
        print(f"\n[QONTEX] [*] Stream online event received for {payload.broadcaster.name}! Starting main script...")
        self.is_live = True
        self.waiter_stop.set()

def wait_for_stream(channel, stop_event):
    waiter = StreamOnlineWaiter(channel)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def run_waiter():
        loop.create_task(waiter.start())
        while not waiter.waiter_stop.is_set() and not stop_event.is_set():
            await asyncio.sleep(0.5)
        await waiter.close()
        
    try:
        loop.run_until_complete(run_waiter())
    except asyncio.CancelledError:
        pass
    return waiter.is_live


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    try:
        refresh_twitch_token(strict=True)
    except TwitchCredentialsError as e:
        print(f"[QONTEX] Twitch credentials error: {e}")
        exit(1)
    config = load_config()

    NICK = config.get("TWITCH_USERNAME")
    CHANNEL = config.get("CHANNEL")
    PASS = os.getenv("TWITCH_TOKEN")

    if not CHANNEL:
        print("[QONTEX] Please set the CHANNEL in config.toml!")
        exit(1)

    log_folder = create_stream_folder(CHANNEL.lstrip("#"), datetime.datetime.now())
    listener = TwitchChatListener(NICK, PASS, CHANNEL, log_folder)
    print(f"[QONTEX] Testing Twitch chat on #{CHANNEL.lstrip('#')}...")
    listener.listen()
