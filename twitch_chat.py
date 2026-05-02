import asyncio
import datetime
import os
import threading

from twitchio.ext import commands

from utils import Message, create_stream_folder, is_likely_question, log_json, log_message, shared_deque, refresh_twitch_token


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
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)

        token = self._normalize_token(password)
        super().__init__(token=token, prefix="!", initial_channels=[self.channel], loop=self._event_loop)

    def _normalize_token(self, token):
        if not token:
            raise ValueError("TWITCH_TOKEN is required for Twitch chat.")
        return token if token.startswith("oauth:") else f"oauth:{token}"

    def is_likely_question(self, message):
        return is_likely_question(message)

    def handle_question(self, msg):
        self.question_queue.append({"user": msg.user, "msg": msg.text, "timestamp": msg.timestamp})
        if self.question_handler:
            threading.Thread(target=self.question_handler, args=(msg,), daemon=True).start()

    def listen(self):
        asyncio.set_event_loop(self._event_loop)
        self._event_loop.run_until_complete(self.start())

    async def event_ready(self):
        print(f"TwitchIO connected as {self.bot_nick}. Joined #{self.channel}.")
        await self.refresh_stream_info()
        asyncio.create_task(self.refresh_stream_info_loop())

    async def event_message(self, message):
        if getattr(message, "echo", False):
            return

        username = getattr(getattr(message, "author", None), "name", None) or "unknown"
        chat_msg = getattr(message, "content", "").strip()
        if not chat_msg:
            return

        msg_time = self.start_time_ref() if self.start_time_ref else datetime.datetime.now().timestamp()
        msg = Message(msg_time, "chat", chat_msg, user=username)

        print(f"\r{msg}")

        log_message(self.log_folder, "chat.log", msg)
        log_json(self.log_folder, "merged.json", msg.to_dict())
        shared_deque.add_message(msg)

        if self.is_likely_question(chat_msg):
            self.handle_question(msg)
            print(f"\r[!] New question detected! (Total in queue: {len(self.question_queue)})")

        await self.handle_commands(message)

    async def refresh_stream_info_loop(self):
        while True:
            await asyncio.sleep(300)
            await self.refresh_stream_info()

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
    import json

    from dotenv import load_dotenv

    load_dotenv()
    refresh_twitch_token()
    with open("config.json", "r") as f:
        config = json.load(f)

    NICK = config.get("TWITCH_USERNAME")
    CHANNEL = config.get("CHANNEL")
    PASS = os.getenv("TWITCH_TOKEN")

    if not CHANNEL:
        print("Please set the CHANNEL in config.json!")
        exit(1)

    log_folder = create_stream_folder(CHANNEL.lstrip("#"), datetime.datetime.now())
    listener = TwitchChatListener(NICK, PASS, CHANNEL, log_folder)
    print(f"Testing Twitch chat on #{CHANNEL.lstrip('#')}...")
    listener.listen()
