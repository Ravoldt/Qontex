import datetime
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional


@dataclass(init=False)
class Message:
    timestamp: float
    type: str
    text: str
    user: Optional[str]
    extra: Dict[str, Any]

    def __init__(self, timestamp, msg_type, text, user=None, **kwargs):
        self.timestamp = float(timestamp or 0.0)
        self.type = msg_type
        self.text = text
        self.user = user
        self.extra = kwargs

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "type": self.type,
            "text": self.text,
            "user": self.user,
            **self.extra,
        }

    def format_timestamp(self):
        total_seconds = max(0, int(self.timestamp))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return "[{}:{:02d}:{:02d}]".format(hours, minutes, seconds)

    def __str__(self):
        ts = self.format_timestamp()
        if self.type == "transcript":
            return "{} {}: {}".format(ts, self.user or "TRANSCRIPT", self.text)
        if self.type == "chat":
            return "{} {}: {}".format(ts, self.user, self.text)
        return "{} {}: {}".format(ts, self.type.upper(), self.text)


class SharedDeque:
    def __init__(self, max_age_seconds=180):
        self.deque = deque()
        self.max_age = max_age_seconds
        self.lock = threading.Lock()

    def add_message(self, message):
        with self.lock:
            self.deque.append(message)
            current_time = message.timestamp
            while self.deque and (current_time - self.deque[0].timestamp) > self.max_age:
                self.deque.popleft()

    def get_recent(self):
        with self.lock:
            return list(self.deque)

    def clear(self):
        with self.lock:
            self.deque.clear()


@dataclass
class StreamConfig:
    channel: str
    source: str
    source_kind: str
    stream_name: str
    game_name: str
    twitch_username: Optional[str] = None
    chat_path: Optional[str] = None
    process_fast: bool = False
    enable_question_detector: bool = True
    question_detector_type: str = "standard"
    enable_qa: bool = True
    enable_qa_chat: bool = True
    enable_qa_transcript: bool = True
    enable_items: bool = False
    enable_visual_context: bool = False
    enable_audio_context: bool = False
    visual_context_max_frames: int = 5
    visual_context_fps: float = 1.0
    audio_context_window: int = 60
    qa_context_window: int = 60
    log_answers_separately: bool = False
    qa_agent: str = "gemini"
    qa_model: Optional[str] = None

    @property
    def is_vod(self):
        return self.source_kind == "vod"

    @property
    def is_livestream(self):
        return self.source_kind == "livestream"

    @property
    def is_local_video(self):
        return self.is_vod and os.path.isfile(self.source)

    @classmethod
    def from_raw(cls, config):
        channel = config.get("CHANNEL")
        if not channel:
            raise ValueError("Please set the CHANNEL in config.toml!")

        source_type = str(config.get("SOURCE_TYPE", "auto")).strip().lower()
        if source_type not in ("auto", "livestream", "live", "vod", "local", "file"):
            raise ValueError("SOURCE_TYPE must be 'auto', 'livestream', or 'vod'.")

        forced_vod = source_type in ("vod", "local", "file")
        forced_live = source_type in ("livestream", "live")
        is_local_file = os.path.isfile(channel)
        is_vod = forced_vod or (source_type == "auto" and is_local_file)

        if is_vod:
            if not is_local_file and not channel.startswith("http"):
                raise ValueError(f"VOD source was not found: '{channel}'")
            source = channel
            twitch_channel = channel
            stream_name = os.path.splitext(os.path.basename(channel.rstrip("/\\")))[0] or channel
            chat_path = config.get("CHAT_FILE") or _default_chat_path(channel)
        else:
            if forced_live and os.path.isfile(channel):
                raise ValueError("SOURCE_TYPE is 'livestream', but CHANNEL points to a local file.")
            if not channel.startswith("http") and ("/" in channel or "\\" in channel or "." in channel):
                raise ValueError(f"Local video file not found, or invalid Twitch channel name: '{channel}'")
            if channel.startswith("http"):
                source = channel.rstrip("/").split("/")[-1]
                twitch_channel = f"#{source}"
            else:
                twitch_channel = channel if channel.startswith("#") else f"#{channel}"
                source = twitch_channel.lstrip("#")
            stream_name = source
            chat_path = config.get("CHAT_FILE") or None

        return cls(
            channel=twitch_channel,
            source=source,
            source_kind="vod" if is_vod else "livestream",
            stream_name=stream_name,
            game_name=config.get("GAME_NAME") or config.get("GAME") or stream_name,
            twitch_username=config.get("TWITCH_USERNAME"),
            chat_path=chat_path,
            process_fast=config.get("PROCESS_FAST", False),
            enable_question_detector=config.get("ENABLE_QUESTION_DETECTOR", True),
            question_detector_type=config.get("QUESTION_DETECTOR_TYPE", "standard"),
            enable_qa=config.get("ENABLE_QA", True),
            enable_qa_chat=config.get("ENABLE_QA_CHAT", True),
            enable_qa_transcript=config.get("ENABLE_QA_TRANSCRIPT", True),
            enable_items=config.get("ENABLE_ITEMS", False),
            enable_visual_context=config.get("ENABLE_VISUAL_CONTEXT", False),
            enable_audio_context=config.get("ENABLE_AUDIO_CONTEXT", False),
            visual_context_max_frames=config.get("VISUAL_CONTEXT_MAX_FRAMES", 5),
            visual_context_fps=config.get("VISUAL_CONTEXT_FPS", 1.0),
            audio_context_window=config.get("AUDIO_CONTEXT_WINDOW", 60),
            qa_context_window=config.get("QA_CONTEXT_WINDOW", 60),
            log_answers_separately=config.get("LOG_ANSWERS_SEPARATELY", False),
            qa_agent=str(config.get("QA_AGENT") or "gemini").strip(),
            qa_model=config.get("QA_MODEL") or None,
        )


@dataclass
class StreamState:
    config: StreamConfig
    log_folder: str
    stream_start: datetime.datetime
    clock_start_wall: float = field(default_factory=time.time)
    timeline: SharedDeque = field(default_factory=SharedDeque)
    video_frames: Deque[Any] = field(default_factory=lambda: deque(maxlen=180))
    audio_chunks: Deque[bytes] = field(default_factory=lambda: deque(maxlen=int(180 / 0.032)))
    agent: Any = None
    chat_listener: Any = None
    listener_thread: Optional[threading.Thread] = None
    capture_thread: Optional[threading.Thread] = None
    capture_stop: Optional[threading.Event] = None
    enable_items: bool = False
    current_video_timestamp: float = 0.0
    chat_processing_timestamp: Optional[float] = None

    def __post_init__(self):
        self.enable_items = self.config.enable_items
        self.configure_video_buffer(self.config.visual_context_fps)
        self.configure_audio_buffer()

    def current_timestamp(self):
        return time.time() - self.clock_start_wall

    def update_stream_start(self, started_at):
        if isinstance(started_at, datetime.datetime):
            self.clock_start_wall = started_at.timestamp()
        else:
            self.clock_start_wall = float(started_at)

    def configure_video_buffer(self, target_fps=None):
        fps = max(0.1, float(target_fps or self.config.visual_context_fps or 1.0))
        self.video_frames = deque(maxlen=max(1, int(self.timeline.max_age * fps)))
        return self.video_frames

    def configure_audio_buffer(self):
        self.audio_chunks = deque(maxlen=max(1, int(self.timeline.max_age / 0.032)))
        return self.audio_chunks

    def add_message(self, message):
        self.timeline.add_message(message)

    def recent_messages(self):
        return self.timeline.get_recent()

    def context_messages(self, timestamp, window):
        return [
            message
            for message in self.timeline.get_recent()
            if abs(float(timestamp) - message.timestamp) <= window
        ]

    def clear_buffers(self):
        self.timeline.clear()
        self.video_frames.clear()
        self.audio_chunks.clear()

    def refresh_config(self, config, log_folder=None):
        self.config = config
        self.enable_items = config.enable_items
        if log_folder is not None:
            self.log_folder = log_folder


def _default_chat_path(source):
    if not source or source.startswith("http"):
        return None
    return os.path.splitext(source)[0] + ".json"
