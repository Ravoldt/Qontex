import os
import json
import re
import subprocess
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from collections import deque
import threading
import time
from transformers import pipeline
import tomllib

_log_lock = threading.Lock()
_config_lock = threading.Lock()
_classifier = None
_config = {}

def _read_toml(path):
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Syntax error in '{path}': {e}")

def load_config(path="config.toml", local_path="local.toml"):
    """Load shared config, then overlay local machine-specific settings."""
    global _config
    config = _read_toml(path)
    if local_path and os.path.exists(local_path):
        config.update(_read_toml(local_path))
    with _config_lock:
        _config = config
    return config

def reload_config(path="config.toml", local_path="local.toml"):
    return load_config(path, local_path)

def get_config_value(key, default=None):
    with _config_lock:
        return _config.get(key, default)

def _log_question_detection(message, msg_type, method, score=None, detail=None):
    if not get_config_value("LOG_QUESTION_DETECTIONS", True):
        return

    source_tag = "[CHAT]" if msg_type == "chat" else "[TRANSCRIPT]"
    detail_text = f" [{detail}]" if detail else ""
    if score is None:
        print(f"\r[!] Detected {source_tag} question via {method}{detail_text}: {message}")
    else:
        print(f"\r[!] Detected {source_tag} question via {method}{detail_text} (Score: {score:.2f}): {message}")

try:
    load_config()
except Exception:
    _config = {}

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
        total_seconds = max(0, int(self.timestamp))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return "[{}:{:02d}:{:02d}]".format(hours, minutes, seconds)

    def __str__(self):
        ts = self.format_timestamp()
        if self.type == "transcript":
            return "{} {}: {}".format(ts, self.user or "TRANSCRIPT", self.text)
        elif self.type == "chat":
            return "{} {}: {}".format(ts, self.user, self.text)
        else:
            return "{} {}: {}".format(ts, self.type.upper(), self.text)

def get_streamer_name(source):
    """Extract streamer name from URL or return as is."""
    if source.startswith("http"):        
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

_json_buffer = []
_buffer_lock = threading.Lock()
_flush_thread = None
_flush_stop_event = threading.Event()

def start_buffer_flusher():
    global _flush_thread
    with _buffer_lock:
        if _flush_thread is None or not _flush_thread.is_alive():
            _flush_thread = threading.Thread(target=_flush_loop, daemon=True)
            _flush_thread.start()

def _flush_loop():
    while not _flush_stop_event.is_set():
        time.sleep(1)
        flush_json_buffer(force=False)

def flush_json_buffer(force=True):
    global _json_buffer
    delay = get_config_value("LOG_BUFFER_DELAY", 20)
    now = time.time()
    to_flush, kept = [], []
    with _buffer_lock:
        for item in _json_buffer:
            if force or (now - item[3] >= delay):
                to_flush.append(item)
            else:
                kept.append(item)
        _json_buffer = kept
    if not to_flush:
        return
    grouped = {}
    for folder, filename, data, _ in to_flush:
        grouped.setdefault((folder, filename), []).append(data)
    
    def safe_ts(x):
        try:
            return float(x.get("timestamp", 0) if x.get("timestamp") is not None else 0)
        except (ValueError, TypeError):
            return 0.0
            
    for (folder, filename), items in grouped.items():
        items.sort(key=safe_ts)
        path = os.path.join(folder, filename)
        with _log_lock:
            with open(path, "a", encoding='utf-8') as f:
                for data in items:
                    json.dump(data, f, ensure_ascii=False)
                    f.write('\n')

def log_json(folder, filename, data, mode='a'):
    """Append data to JSON buffer to be sorted and written later."""
    if mode != 'a':
        path = os.path.join(folder, filename)
        with _log_lock:
            with open(path, mode, encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
        return
        
    with _buffer_lock:
        _json_buffer.append((folder, filename, data, time.time()))
    if _flush_thread is None:
        start_buffer_flusher()

def preload_classifier():
    global _classifier
    if _classifier is None:
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading Question Classifier into {'VRAM' if device == 0 else 'RAM'}...")        
        torch_dtype = torch.float16 if device == 0 else torch.float32
        _classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3",
            device=device,
            torch_dtype=torch_dtype)

def is_likely_question(message, msg_type="chat"):
    """Return True when a chat or transcript message looks like a question."""
    if not get_config_value("ENABLE_QUESTION_DETECTOR", True):
        return False
        
    global _classifier
    
    clean_message = message.strip()
    
    # Strip emotes from chat messages to avoid confusing the AI classifier
    if msg_type == "chat":
        clean_message = re.sub(r'\b[a-z]+[A-Z][A-Za-z]*\b', '', clean_message)
        clean_message = re.sub(r'\s+', ' ', clean_message).strip()
    
    if not clean_message:
        return False
        
    message_lower = clean_message.lower()

    # Statements that look like questions but usually aren't
    looks_like_question_starters = (
        "what a ", "what an ", "how nice", "how cool", "how cute", "how i",
        "where i", "when i", "when you", "where you", "why i", "why you"
    )
    if any(
        re.search(r'(?<![a-z0-9]){}\b'.format(re.escape(starter.strip())), message_lower)
        for starter in looks_like_question_starters
    ):
        return False

    # Fast Exits
    #if message_lower.endswith("?"):
    #    _log_question_detection(message, msg_type, "question mark")
    #    return True
    #if message_lower.endswith("..."):
    #    return False

    # Common question starters
    question_starters = (
        #"who ", "what ", "where ", "when ", "why ", "how ", 
        "is there ", "are there ", "can i ", "do you ", "did you ", "have you ", "does he ", "what's ", "who's ", "where's ", "when's ", "why's ", "how's ",
        "is it ", "are they ", "could i ", "i wonder if ", "anyone know ", "can anyone ", "does anyone ", "would anyone ", "should i ", "am i "
    )
    matched_starter = next(
        (
            starter.strip()
            for starter in question_starters
            if re.search(r'(?<![a-z0-9]){}\b'.format(re.escape(starter.strip())), message_lower)
        ),
        None,
    )
    if matched_starter:
        _log_question_detection(message, msg_type, "starter phrase", detail=matched_starter)
        return True

    words = message_lower.split()

    # Length filter for chat only
    if msg_type == "chat":        
        filter_short = get_config_value("FILTER_SHORT_QUESTIONS", True)
        threshold = get_config_value("SHORT_QUESTION_THRESHOLD", 2)
        if filter_short and len(words) <= threshold:
            return False

        # The Backseater Bypass, Drop long paragraphs without question marks
        if len(words) > 15 and "?" not in message:
            return False

    # AI Classification
    if _classifier is None:
        preload_classifier()

    # classify on answerability
    labels = ["information request", "reaction or exclamation", "statement"]
    result = _classifier(clean_message, candidate_labels=labels)
    
    top_label = result["labels"][0]
    question_score = result["scores"][0]
    is_question = top_label == "information request" and question_score >= 0.60
    
    if is_question:
        _log_question_detection(message, msg_type, "AI classifier", question_score)
        
    return is_question

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

    def clear(self):
        with self.lock:
            self.deque.clear()

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
        "scope": "user:read:chat" #user:write:chat is not needed currently
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
