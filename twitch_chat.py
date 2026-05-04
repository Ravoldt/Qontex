import asyncio
import datetime
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
