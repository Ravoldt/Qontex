import json
import os
import re
import threading

import cv2
from google import genai
import PIL.Image

from utils import Message, log_json, shared_deque, get_streamer_name


class GeminiAgent:
    def __init__(self, api_key, log_folder, start_time_ref=None, game_name=None, qa_context_window=60, enable_visual_context=False, streamer_name="the streamer", log_answers_separately=False):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"
        self.log_folder = log_folder
        self.start_time_ref = start_time_ref
        self.game_name = game_name or "the game being played on stream"
        self.streamer_name = streamer_name
        self.qa_context_window = qa_context_window
        self.enable_visual_context = enable_visual_context
        self.log_answers_separately = log_answers_separately
        self.item_duplicate_window_seconds = 240
        self._item_lock = threading.Lock()
        self._seen_item_events = []
        self._load_seen_item_events()

    def _current_timestamp(self):
        return self.start_time_ref() if self.start_time_ref else 0

    def set_game_name(self, game_name):
        if game_name:
            self.game_name = game_name

    def ask_gemini(self, username: str, question: str, source_type: str = "chat", timestamp: float = None, video_frames_deque = None):
        """
        Sends a user's question to the Gemini model along with relevant stream context to generate an answer.

        Args:
            username: The name of the user asking the question.
            question: The question text to be answered.
            source_type: The origin of the question (e.g., "chat" or "transcript").
            timestamp: The stream time the question was asked. Defaults to the current stream time.
            video_frames_deque: A deque containing recent video frames for visual context.
        """
        
        # Add a 1ms offset so it stably sorts immediately after the original question
        msg_time = (timestamp + 0.001) if timestamp is not None else self._current_timestamp()
        
        all_messages = shared_deque.get_recent()
        context_msgs = [
            str(m) for m in all_messages
            if abs(msg_time - m.timestamp) <= self.qa_context_window
        ]
        context_str = "\n".join(context_msgs) if context_msgs else "No recent context available."

        prompt = f"""
        Answer the target question from {self.streamer_name}'s {self.game_name} stream.
        If the question doesn't have an objective answer return exactly: NO_ANSWER
        If answering, write one concise sentence or short paragraph that can be understood without seeing the original question.
        Do not include labels, markdown, apologies, caveats, or additional commentary.

Context:
{context_str}

Target Question from '{username}':
{question}"""

        contents = [prompt]

        if self.enable_visual_context and video_frames_deque and len(video_frames_deque) > 0:
            frames = list(video_frames_deque)[-int(self.qa_context_window):]
            step = max(1, len(frames) // 5)
            for frame in frames[::step][:5]:
                rgb_f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(rgb_f)
                contents.append(img)

        try:
            response = self.client.models.generate_content(model=self.model_name, contents=contents)
            answer = response.text.strip()            
            if not answer or answer == "NO_ANSWER":
                return None

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
            if self.log_answers_separately:
                log_json(self.log_folder, "answered_questions.json", gemini_msg.to_dict())
            return answer
        except Exception as e:
            print(f"\nGemini API Error: {e}\n")
            return None

    def direct_ask(self, question, timestamp=None, video_frames_deque=None):
        # Add a 1ms offset for stable sorting
        msg_time = (timestamp + 0.001) if timestamp is not None else self._current_timestamp()
        
        all_messages = shared_deque.get_recent()
        context_msgs = [
            str(m) for m in all_messages
            if abs(msg_time - m.timestamp) <= self.qa_context_window
        ]
        context_str = "\n".join(context_msgs) if context_msgs else "No recent context available."

        prompt = f"""
        One of {self.streamer_name}'s viewers has a question: {question}.
        Answer the question. Below is recent context from the stream that may help you answer.

Context:
{context_str}
"""

        contents = [prompt]

        if self.enable_visual_context and video_frames_deque and len(video_frames_deque) > 0:
            frames = list(video_frames_deque)
            step = max(1, len(frames) // 5)
            for frame in frames[::step][:5]:
                rgb_f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(rgb_f)
                contents.append(img)

        try:
            response = self.client.models.generate_content(model=self.model_name, contents=contents)
            answer = response.text.strip()
            
            print(f"\n[Gemini Console Response]:\n{answer}\n")
            
            gemini_msg = Message(
                msg_time,
                "gemini",
                answer,
                user="GEMINI",
                question=question,
                question_user="Console",
                question_source="console",
            )
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

        if self.enable_visual_context and video_frames_deque and len(video_frames_deque) > 0:
            frames = list(video_frames_deque)
            step = max(1, len(frames) // 5)
            for frame in frames[::step][:5]:
                rgb_f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(rgb_f)
                contents.append(img)

        try:
            response = self.client.models.generate_content(model=self.model_name, contents=contents)
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
