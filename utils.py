import os
import json
import subprocess
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from collections import deque
import threading

_log_lock = threading.Lock()

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
    """Create and return path to logs/streamer_name/YYYY-MM-DD/."""
    if start_time is None:
        start_time = datetime.now()
    date_str = start_time.strftime("%Y-%m-%d")
    path = os.path.join("logs", streamer_name, date_str)
    os.makedirs(path, exist_ok=True)
    return path

def log_message(folder, filename, message, mode='a'):
    """Append message to file in folder."""
    path = os.path.join(folder, filename)
    with _log_lock:
        with open(path, mode, encoding='utf-8') as f:
            f.write(str(message) + '\n')

def log_json(folder, filename, data, mode='a'):
    """Append data to JSON file (for merged timeline, list of dicts)."""
    path = os.path.join(folder, filename)
    with _log_lock:
        with open(path, mode, encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

def is_likely_question(message):
    """Return True when a chat or transcript message looks like a question."""
    message = message.lower().strip()
    if message.endswith("?"):
        return True
    question_starters = [
        "who", "what", "where", "when", "why", "how", "can you", "is there",
        "do you", "does it", "are there", "could you", "would you", "should i",
        "will it", "can i", "should we", "is this", "is that", "did he",
        "did she", "did they", "does this", "does that", "i'm curious",
        "anyone know", "does anyone"
    ]
    return any(message.startswith(q) for q in question_starters)

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

def refresh_twitch_token():
    """Validate the current Twitch token and refresh it if expired."""
    token = os.getenv("TWITCH_TOKEN")
    client_id = os.getenv("TWITCH_CLIENT_ID")
    client_secret = os.getenv("TWITCH_CLIENT_SECRET")
    refresh_token = os.getenv("TWITCH_REFRESH_TOKEN")

    clean_token = token.replace("oauth:", "") if token else ""

    if clean_token:
        req = urllib.request.Request("https://id.twitch.tv/oauth2/validate")
        req.add_header("Authorization", f"OAuth {clean_token}")
        try:
            with urllib.request.urlopen(req) as response:
                if response.getcode() == 200:
                    return token
        except urllib.error.HTTPError:
            pass  # Token is invalid, proceed to refresh

    if not all([client_id, client_secret, refresh_token]):
        print("[!] Twitch token may be expired. Add TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET, and TWITCH_REFRESH_TOKEN to .env to enable auto-refresh.")
        return token

    print("[*] Twitch token expired. Attempting to refresh...")
    data = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "scope": "chat:read"
    }).encode("utf-8")

    req = urllib.request.Request("https://id.twitch.tv/oauth2/token", data=data)
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            new_access_token = result.get("access_token")
            new_refresh_token = result.get("refresh_token")

            if new_access_token:
                try:
                    from dotenv import set_key
                    env_path = os.path.join(os.getcwd(), ".env")
                    set_key(env_path, "TWITCH_TOKEN", new_access_token)
                    if new_refresh_token:
                        set_key(env_path, "TWITCH_REFRESH_TOKEN", new_refresh_token)
                except Exception as e:
                    print(f"[!] Could not update .env automatically: {e}")

                os.environ["TWITCH_TOKEN"] = new_access_token
                if new_refresh_token:
                    os.environ["TWITCH_REFRESH_TOKEN"] = new_refresh_token
                print("[*] Twitch token successfully refreshed!")
                return new_access_token
    except Exception as e:
        print(f"[!] Failed to refresh Twitch token: {e}")

    return token
