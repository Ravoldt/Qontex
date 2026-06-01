import io
import json
import os
import re
import threading
import wave

import cv2
from google import genai
from google.genai import types
import PIL.Image

from qa_agent import BaseQAAgent
from stream_state import Message
from utils import log_json, shared_deque


class GeminiAgent(BaseQAAgent):
    def __init__(
        self,
        api_key,
        log_folder=None,
        start_time_ref=None,
        game_name=None,
        qa_context_window=60,
        audio_context_window=60,
        visual_context_max_frames=5,
        visual_context_fps=1.0,
        enable_visual_context=False,
        enable_audio_context=False,
        streamer_name="the streamer",
        log_answers_separately=False,
        state=None,
        model_name=None,
    ):
        self.client = genai.Client(api_key=api_key)
        self.state = state
        self.model_name = model_name or "gemini-2.5-pro"
        self.log_folder = log_folder
        self.start_time_ref = start_time_ref
        self.game_name = game_name or "the game being played on stream"
        self.streamer_name = streamer_name
        self.qa_context_window = qa_context_window
        self.enable_visual_context = enable_visual_context
        self.enable_audio_context = enable_audio_context
        self.audio_context_window = audio_context_window
        self.visual_context_max_frames = visual_context_max_frames
        self.visual_context_fps = visual_context_fps
        self.log_answers_separately = log_answers_separately
        self.item_duplicate_window_seconds = 240
        self._item_lock = threading.Lock()
        self._seen_item_events = []
        self.refresh_from_state()
        self._load_seen_item_events()

    def refresh_from_state(self):
        if not self.state:
            return

        config = self.state.config
        self.log_folder = self.state.log_folder
        self.game_name = config.game_name or self.game_name
        self.streamer_name = config.stream_name or self.streamer_name
        self.qa_context_window = config.qa_context_window
        self.enable_visual_context = config.enable_visual_context
        self.enable_audio_context = config.enable_audio_context
        self.audio_context_window = config.audio_context_window
        self.visual_context_max_frames = config.visual_context_max_frames
        self.visual_context_fps = config.visual_context_fps
        self.log_answers_separately = config.log_answers_separately
        if config.qa_model:
            self.model_name = config.qa_model

    def _current_timestamp(self):
        if self.state:
            return self.state.current_timestamp()
        return self.start_time_ref() if self.start_time_ref else 0

    def set_game_name(self, game_name):
        if game_name:
            self.game_name = game_name

    def answer_question(self, message):
        return self.ask_gemini(
            message.user,
            message.text,
            source_type=message.type,
            timestamp=message.timestamp,
        )

    def ask_gemini(
        self,
        username: str,
        question: str,
        source_type: str = "chat",
        timestamp: float = None,
        video_frames_deque=None,
        audio_chunks_deque=None,
    ):
        """Answer a detected stream question with recent context from StreamState."""
        msg_time = (timestamp + 0.001) if timestamp is not None else self._current_timestamp()
        context_str = self._context_string(msg_time, self.qa_context_window)

        prompt = f"""
        Answer the target question from {self.streamer_name}'s stream.
        Please search for an accurate and up-to-date answer, if the answer cannot be found in the provided context.
        Use the provided stream context, video frames, and audio context as supplemental information to help understand what the user is referring to.
        If the question doesn't have an objective answer return exactly: NO_ANSWER
        If answering, write one concise sentence or short paragraph that can be understood without seeing the original question.
        Do not include labels, markdown, apologies, caveats, or additional commentary.

Context:
{context_str}

Target Question from '{username}':
{question}"""

        contents = [prompt]
        self._append_media_context(contents, video_frames_deque, audio_chunks_deque, slice_to_context_window=True)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                ),
            )
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
            self._log_answer(gemini_msg)
            return answer
        except Exception as e:
            print(f"\n[QONTEX] Gemini API Error: {e}\n")
            return None

    def direct_ask(self, question, timestamp=None, video_frames_deque=None, audio_chunks_deque=None):
        msg_time = (timestamp + 0.001) if timestamp is not None else self._current_timestamp()
        context_str = self._context_string(msg_time, self.qa_context_window)

        prompt = f"""
        One of {self.streamer_name}'s viewers has a question: {question}.
        Answer the question by using Google Search to find the most accurate and up-to-date answer.
        Below is recent context from the stream. Use this context along with any provided video or audio as supplemental information to help answer the question.

Context:
{context_str}
"""

        contents = [prompt]
        initial_count = len(contents)
        self._append_media_context(contents, video_frames_deque, audio_chunks_deque, slice_to_context_window=False)
        attached_frames = max(0, len(contents) - initial_count)
        if attached_frames and self.enable_visual_context:
            print(f"[QONTEX] [*] Attached {attached_frames} media parts to Gemini direct ask.")

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}],
                ),
            )
            answer = response.text.strip()

            print(f"\n[QONTEX] [Gemini Console Response]:\n{answer}\n")

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
            print(f"\n[QONTEX] Gemini API Error: {e}\n")
            return None

    def process_items(self, video_frames_deque=None, audio_chunks_deque=None):
        messages = self._recent_messages()
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
        self._append_media_context(contents, video_frames_deque, audio_chunks_deque, slice_to_context_window=False)

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
            print(f"[QONTEX] Gemini item processing error: {e}")

    def _context_string(self, timestamp, window):
        context_msgs = [str(m) for m in self._context_messages(timestamp, window)]
        return "\n".join(context_msgs) if context_msgs else "No recent context available."

    def _context_messages(self, timestamp, window):
        if self.state:
            return self.state.context_messages(timestamp, window)
        return [
            message
            for message in shared_deque.get_recent()
            if abs(float(timestamp) - message.timestamp) <= window
        ]

    def _recent_messages(self):
        if self.state:
            return self.state.recent_messages()
        return shared_deque.get_recent()

    def _append_media_context(self, contents, video_frames_deque=None, audio_chunks_deque=None, slice_to_context_window=False):
        video_frames_deque = video_frames_deque if video_frames_deque is not None else self._state_video_frames()
        audio_chunks_deque = audio_chunks_deque if audio_chunks_deque is not None else self._state_audio_chunks()

        if self.enable_visual_context and video_frames_deque is not None and len(video_frames_deque) > 0:
            frames = list(video_frames_deque)
            if slice_to_context_window:
                frames_to_slice = max(1, int(self.qa_context_window * self.visual_context_fps))
                frames = frames[-frames_to_slice:]
            step = max(1, len(frames) // max(1, self.visual_context_max_frames))
            for frame in frames[::step][:self.visual_context_max_frames]:
                rgb_f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(rgb_f)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="JPEG")
                contents.append(types.Part.from_bytes(data=img_byte_arr.getvalue(), mime_type="image/jpeg"))

        if self.enable_audio_context and audio_chunks_deque is not None and len(audio_chunks_deque) > 0:
            num_chunks = max(1, int(self.audio_context_window / 0.032))
            recent_audio = list(audio_chunks_deque)[-num_chunks:]
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"".join(recent_audio))
            contents.append(types.Part.from_bytes(data=wav_io.getvalue(), mime_type="audio/wav"))

    def _state_video_frames(self):
        return self.state.video_frames if self.state else None

    def _state_audio_chunks(self):
        return self.state.audio_chunks if self.state else None

    def _log_answer(self, message):
        log_json(self.log_folder, "merged.json", message.to_dict())
        if self.log_answers_separately:
            log_json(self.log_folder, "answered_questions.json", message.to_dict())

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
        if not self.log_folder:
            return

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
            print(f"[QONTEX] Gemini item de-duplication preload failed: {e}")

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
