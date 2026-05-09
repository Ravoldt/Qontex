import asyncio
import datetime
import json
import os
import time
import threading
import twitchio
from twitchio.ext import commands

from utils import Message, create_stream_folder, get_config_value, is_likely_question, load_config, log_json, log_message, shared_deque, refresh_twitch_token


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
    ):
        self.bot_nick = nick
        self.channel = channel.lstrip("#").lower()
        self.log_folder = log_folder
        self.start_time_ref = start_time_ref
        self.question_handler = question_handler
        self.category_handler = category_handler
        self.stream_start_handler = stream_start_handler
        self.question_queue = []
        self.stream_category = None
        self._fallback_start = time.time()
        self._info_loop_started = False
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
        if not token:
            raise ValueError("TWITCH_TOKEN is required for Twitch chat.")
        token = token.strip()
        if token.startswith("oauth:"):
            return token.removeprefix("oauth:")
        if token.startswith("OAuth "):
            return token.removeprefix("OAuth ")
        if token.startswith("Bearer "):
            return token.removeprefix("Bearer ")
        return token

    async def setup_hook(self):
        refresh_token = os.getenv("TWITCH_REFRESH_TOKEN")
        if not refresh_token:
            raise ValueError("TWITCH_REFRESH_TOKEN is required for Twitch EventSub chat.")

        validated = await self.add_token(self.twitch_token, refresh_token)
        client_id = os.getenv("TWITCH_CLIENT_ID")
        if validated.client_id != client_id:
            raise ValueError(
                "TWITCH_TOKEN was issued for client ID "
                f"{validated.client_id}, but TWITCH_CLIENT_ID is {client_id}."
            )

        if validated.user_id != self.bot_id:
            raise ValueError(
                "TWITCH_TOKEN belongs to user ID "
                f"{validated.user_id}, but TWITCH_BOT_ID is {self.bot_id}."
            )

        if "user:read:chat" not in validated.scopes:
            raise ValueError("TWITCH_TOKEN must include the user:read:chat scope.")

    def is_likely_question(self, message, msg_type="chat"):
        return is_likely_question(message, msg_type)

    def handle_question(self, msg):
        self.question_queue.append({"user": msg.user, "msg": msg.text, "timestamp": msg.timestamp})
        if len(self.question_queue) > 20:
            self.question_queue.pop(0)
        if self.question_handler:
            threading.Thread(target=self.question_handler, args=(msg,), daemon=True).start()

    def listen(self):
        asyncio.set_event_loop(self._event_loop)
        self._event_loop.run_until_complete(self.start())

    def stop(self):
        if self._event_loop.is_running():
            self._event_loop.call_soon_threadsafe(lambda: asyncio.create_task(self.close()))

    async def event_ready(self):
        print(f"Logged in to Twitch as | {self.bot_nick}")

        # 1. Fetch the streamer's user info to get their numeric Broadcaster ID
        users = await self.fetch_users(logins=[self.channel])
        if not users:
            print(f"Could not find Twitch channel: {self.channel}")
            return
        broadcaster_id = users[0].id

        subscription = twitchio.eventsub.ChatMessageSubscription(
            broadcaster_user_id=broadcaster_id,
            user_id=self.bot_id
        )

        await self.subscribe_websocket(payload=subscription)
        print(f"Successfully subscribed to live chat for {self.channel}!")
        
        if not self._info_loop_started:
            self._info_loop_started = True
            asyncio.create_task(self.refresh_stream_info_loop())

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
        shared_deque.add_message(msg)

        should_detect_question = self.question_handler or get_config_value("LOG_QUESTION_DETECTIONS", True)
        if should_detect_question and self.is_likely_question(msg.text, msg.type):
            self.handle_question(msg)

    async def refresh_stream_info_loop(self):
        while True:
            await self.refresh_stream_info()
            await asyncio.sleep(300)

    async def refresh_stream_info(self):
        category, started_at = await self.fetch_stream_info()
        if category and category != self.stream_category:
            self.stream_category = category
            if self.category_handler:
                self.category_handler(category)
            print(f"\rTwitch category: {category}")
        
        if started_at and self.stream_start_handler:
            self.stream_start_handler(started_at.timestamp())
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

            channel_info = await self.fetch_channel(self.channel)
            return getattr(channel_info, "game_name", None), None
        except Exception as e:
            print(f"\r[!] Twitch stream info lookup failed: {e}")
        return None, None


class LocalChatListener:
    def __init__(self, json_path, log_folder, question_handler=None, process_fast=False):
        self.json_path = json_path
        self.log_folder = log_folder
        self.question_handler = question_handler
        self.process_fast = process_fast
        self.messages = []
        self._stop_event = threading.Event()
        self.question_queue = []
        self.streamer_name = None
        self.streamer_login = None
        self.stream_start_date = None
        self._load_messages()
        
        import streamcapture
        if self.messages:
            streamcapture.chat_processing_timestamp = self.messages[0]["timestamp"]
        else:
            streamcapture.chat_processing_timestamp = float('inf')

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
            print(f"Error loading local chat JSON: {e}")

    def listen(self):
        import streamcapture
        print(f"Loaded {len(self.messages)} chat messages from {self.json_path}")
        msg_idx = 0
        
        try:
            while not self._stop_event.is_set() and msg_idx < len(self.messages):
                target_ts = self.messages[msg_idx]["timestamp"]
                streamcapture.chat_processing_timestamp = target_ts
                
                current_ts = streamcapture.current_video_timestamp
                
                if current_ts < target_ts:
                    time.sleep(0.01)
                    continue
                self._process_message(self.messages[msg_idx])
                msg_idx += 1

                if msg_idx < len(self.messages):
                    streamcapture.chat_processing_timestamp = self.messages[msg_idx]["timestamp"]
                else:
                    streamcapture.chat_processing_timestamp = float('inf')
        finally:
            streamcapture.chat_processing_timestamp = float('inf')

    def _process_message(self, msg_data):
        msg = Message(msg_data["timestamp"], "chat", msg_data["text"], user=msg_data["user"])
        print(f"\r{msg}")
        log_message(self.log_folder, "chat.log", msg)
        log_json(self.log_folder, "merged.json", msg.to_dict())
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


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    refresh_twitch_token()
    config = load_config()

    NICK = config.get("TWITCH_USERNAME")
    CHANNEL = config.get("CHANNEL")
    PASS = os.getenv("TWITCH_TOKEN")

    if not CHANNEL:
        print("Please set the CHANNEL in config.toml!")
        exit(1)

    log_folder = create_stream_folder(CHANNEL.lstrip("#"), datetime.datetime.now())
    listener = TwitchChatListener(NICK, PASS, CHANNEL, log_folder)
    print(f"Testing Twitch chat on #{CHANNEL.lstrip('#')}...")
    listener.listen()
