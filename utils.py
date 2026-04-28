import os
import json
import subprocess
from datetime import datetime, timedelta
from collections import deque
import threading

class Message:
    def __init__(self, timestamp, msg_type, text, user=None, **kwargs):
        self.timestamp = timestamp  # float seconds
        self.type = msg_type  # "transcript" or "chat"
        self.text = text
        self.user = user  # for chat, the username; for transcript, "streamer"
        self.extra = kwargs  # any additional data

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "type": self.type,
            "text": self.text,
            "user": self.user,
            **self.extra
        }

    def format_timestamp(self):
        """Format timestamp as [h:mm:ss]"""
        td = timedelta(seconds=int(self.timestamp))
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return "[{}:{:02d}:{:02d}]".format(hours, minutes, seconds)

    def __str__(self):
        ts = self.format_timestamp()
        if self.type == "transcript":
            return "{} TRANSCRIPT: {}".format(ts, self.text)
        elif self.type == "chat":
            return "{} {}: {}".format(ts, self.user, self.text)
        else:
            return "{} {}: {}".format(ts, self.type.upper(), self.text)

def get_streamer_name(source):
    """Extract streamer name from URL or return as is."""
    if source.startswith("http"):
        # e.g., https://www.twitch.tv/babylon340 -> babylon340
        parts = source.rstrip('/').split('/')
        return parts[-1]
    else:
        return source  # assume it's the name

def create_stream_folder(streamer_name, start_time=None):
    """Create and return path to logs/streamer_name/start_datetime/"""
    if start_time is None:
        start_time = datetime.now()
    start_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join("logs", streamer_name, start_str)
    os.makedirs(path, exist_ok=True)
    return path

def log_message(folder, filename, message, mode='a'):
    """Append message to file in folder."""
    path = os.path.join(folder, filename)
    with open(path, mode, encoding='utf-8') as f:
        f.write(str(message) + '\n')

def log_json(folder, filename, data, mode='a'):
    """Append data to JSON file (for merged timeline, list of dicts)."""
    path = os.path.join(folder, filename)
    with open(path, mode, encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')

def log_start_stop(folder, action, uptime=None):
    """Log start or stop with datetime and optional uptime."""
    now = datetime.now().isoformat()
    if action == "start":
        msg = "Logging started at {}".format(now)
    elif action == "stop":
        uptime_str = " - uptime: {}".format(timedelta(seconds=int(uptime))) if uptime else ""
        msg = "Logging stopped at {}{}".format(now, uptime_str)
    else:
        msg = "Logging {} at {}".format(action, now)
    log_message(folder, "session.log", msg)

class SharedDeque:
    def __init__(self, max_age_seconds=180):  # 3 minutes
        self.deque = deque()
        self.max_age = max_age_seconds
        self.lock = threading.Lock()

    def add_message(self, message):
        with self.lock:
            self.deque.append(message)
            # Remove old messages
            current_time = message.timestamp
            while self.deque and (current_time - self.deque[0].timestamp) > self.max_age:
                self.deque.popleft()

    def get_recent(self):
        with self.lock:
            return list(self.deque)

# Global shared deque
shared_deque = SharedDeque()

def cut_video_segment(input_path, output_path, start_time, duration):
    """Cut a segment from video using ffmpeg."""
    command = [
        "ffmpeg", "-i", input_path, "-ss", str(start_time), "-t", str(duration),
        "-c", "copy", output_path, "-y"
    ]
    try:
        subprocess.run(command, check=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print("Error cutting video: {}".format(e))