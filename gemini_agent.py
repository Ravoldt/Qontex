import json
import os
import re
import threading

import cv2
import google.generativeai as genai
import PIL.Image

from utils import Message, log_json, shared_deque


class GeminiAgent:
    def __init__(self, api_key, log_folder, start_time_ref=None, game_name=None):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-3-flash-preview")
        self.log_folder = log_folder
        self.start_time_ref = start_time_ref
        self.game_name = game_name or "the game being played on stream"
        self.item_duplicate_window_seconds = 240
        self._item_lock = threading.Lock()
        self._seen_item_events = []
        self._load_seen_item_events()

    def _current_timestamp(self):
        return self.start_time_ref() if self.start_time_ref else 0

    def set_game_name(self, game_name):
        if game_name:
            self.game_name = game_name

    def ask_gemini(self, username, question, source_type="chat", timestamp=None):
        prompt = f"""
The {source_type} message from '{username}' contains this question:
"{question}"

The stream is about {self.game_name}.

Answer only if the question is about the game being played on stream or has an objective, factual answer.
If the question is subjective, personal, rhetorical, or cannot be answered from general factual knowledge, return exactly:
NO_ANSWER

If answering, write one concise sentence or short paragraph that can be understood without seeing the original question.
Do not include labels, markdown, apologies, caveats, or additional commentary.
"""
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            if not answer or answer == "NO_ANSWER":
                return None

            msg_time = timestamp if timestamp is not None else self._current_timestamp()
            gemini_msg = Message(
                msg_time,
                "gemini",
                answer,
                user="GEMINI",
                question=question,
                question_user=username,
                question_source=source_type,
            )
            print(gemini_msg)
            log_json(self.log_folder, "merged.json", gemini_msg.to_dict())
            return answer
        except Exception as e:
            print(f"\nGemini API Error: {e}\n")
            return None

    def process_items(self, video_frames_deque=None):
        messages = shared_deque.get_recent()
        if not messages:
            return

        timestamp = self._current_timestamp()
        context = "\n".join(str(msg) for msg in messages)
        prompt = f"""
Based on this 3-minute context of transcript and chat from a stream, identify items the streamer collected.
Compare against known {self.game_name} items and reject hallucinations or uncertain references.
Return only a valid JSON object with this shape:
{{
  "timestamp": {timestamp},
  "items": [
    {{
      "name": "item name",
      "timestamp": 123.0,
      "evidence": "short evidence from the context"
    }}
  ]
}}
The item timestamp must be the earliest timestamp in the supplied context that supports the collection event.
Use an empty items array if no collected items are confirmed.
Do not wrap the JSON in markdown.

Context:
{context}
"""

        contents = [prompt]

        if video_frames_deque and len(video_frames_deque) > 0:
            frames = list(video_frames_deque)
            step = max(1, len(frames) // 5)
            for frame in frames[::step][:5]:
                rgb_f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(rgb_f)
                contents.append(img)

        try:
            response = self.model.generate_content(contents)
            data = self._parse_json_object(response.text)
            data.setdefault("timestamp", timestamp)
            data.setdefault("items", [])
            data["items"] = self._filter_new_item_events(data["items"], timestamp)
            if not data["items"]:
                return
            log_json(self.log_folder, "collected_items.json", data)
        except Exception as e:
            print(f"Gemini item processing error: {e}")

    def _parse_json_object(self, text):
        raw = text.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        return json.loads(raw)

    def _load_seen_item_events(self):
        path = os.path.join(self.log_folder, "collected_items.json")
        if not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    for item in data.get("items", []):
                        self._remember_item_event(item, data.get("timestamp"))
        except Exception as e:
            print(f"Gemini item de-duplication preload failed: {e}")

    def _filter_new_item_events(self, items, analysis_timestamp):
        new_items = []
        with self._item_lock:
            for item in items:
                if not isinstance(item, dict):
                    continue
                if self._is_duplicate_item_event(item):
                    continue
                self._remember_item_event(item, analysis_timestamp)
                new_items.append(item)
        return new_items

    def _is_duplicate_item_event(self, item):
        normalized_name = self._normalize_item_name(item.get("name", ""))
        event_timestamp = self._item_timestamp(item)
        if not normalized_name:
            return True

        for seen_name, seen_timestamp in self._seen_item_events:
            if seen_name != normalized_name:
                continue
            if seen_timestamp is None or event_timestamp is None:
                return True
            if abs(seen_timestamp - event_timestamp) <= self.item_duplicate_window_seconds:
                return True
        return False

    def _remember_item_event(self, item, fallback_timestamp=None):
        normalized_name = self._normalize_item_name(item.get("name", ""))
        if not normalized_name:
            return

        event_timestamp = self._item_timestamp(item)
        if event_timestamp is None:
            event_timestamp = self._coerce_float(fallback_timestamp)
        self._seen_item_events.append((normalized_name, event_timestamp))

    def _normalize_item_name(self, name):
        return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()

    def _item_timestamp(self, item):
        for key in ("timestamp", "source_timestamp", "event_timestamp"):
            value = self._coerce_float(item.get(key))
            if value is not None:
                return value
        return None

    def _coerce_float(self, value):
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
