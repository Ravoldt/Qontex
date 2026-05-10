#!/usr/bin/env python
"""
Standalone question detection pipeline for JSON, JSONL, and CSV datasets.

Examples:
    python question_detection_pipeline.py chat.json --output detections.jsonl
    python question_detection_pipeline.py transcript.csv --text-column text --type transcript --output detections.csv
    python question_detection_pipeline.py merged.json --include-non-questions --no-classifier
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


TEXT_FIELDS = ("text", "message.body", "message", "body", "content", "msg", "comment")
USER_FIELDS = ("user", "username", "commenter.name", "author", "author.name", "display_name", "name")
TIMESTAMP_FIELDS = (
    "timestamp",
    "content_offset_seconds",
    "time",
    "offset",
    "seconds",
    "start",
    "start_time",
)
TYPE_FIELDS = ("type", "msg_type", "source_type", "source")
KNOWN_CONTAINERS = ("comments", "messages", "data", "items", "records", "rows", "entries")


@dataclass
class DetectionConfig:
    enable_detector: bool = True
    use_classifier: bool = True
    filter_short_questions: bool = False
    short_question_threshold: int = 2
    classifier_threshold: float = 0.60
    long_chat_word_limit: int = 15
    strip_chat_emotes: bool = True


@dataclass
class MessageRecord:
    index: int
    text: str
    msg_type: str = "chat"
    user: Optional[str] = None
    timestamp: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class DetectionResult:
    index: int
    is_question: bool
    text: str
    msg_type: str
    user: Optional[str] = None
    timestamp: Optional[float] = None
    method: Optional[str] = None
    score: Optional[float] = None
    detail: Optional[str] = None
    clean_text: Optional[str] = None
    reason: Optional[str] = None


class QuestionDetector:
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self._classifier = None

    def detect(self, message: str, msg_type: str = "chat") -> DetectionResult:
        if not self.config.enable_detector:
            return self._result(False, message, msg_type, reason="detector disabled")

        clean_message = message.strip()
        if msg_type == "chat" and self.config.strip_chat_emotes:
            clean_message = re.sub(r"\b[a-z]+[A-Z][A-Za-z]*\b", "", clean_message)
            clean_message = re.sub(r"\s+", " ", clean_message).strip()

        if not clean_message:
            return self._result(False, message, msg_type, clean_text=clean_message, reason="empty after cleanup")

        message_lower = clean_message.lower()
        words = message_lower.split()

        if msg_type == "chat":
            if self.config.filter_short_questions and len(words) <= self.config.short_question_threshold:
                return self._result(
                    False,
                    message,
                    msg_type,
                    clean_text=clean_message,
                    reason="short chat message filtered",
                )

            if len(words) > self.config.long_chat_word_limit and "?" not in message:
                return self._result(
                    False,
                    message,
                    msg_type,
                    clean_text=clean_message,
                    reason="long chat message without question mark",
                )

        false_question_starters = (
            "what a ",
            "what an ",
            "how nice",
            "how cool",
            "how cute",
            "how i",
            "where i",
            "when i",
            "when you",
            "where you",
            "why i",
            "why you",
        )
        if any(self._has_phrase(message_lower, starter.strip()) for starter in false_question_starters):
            return self._result(
                False,
                message,
                msg_type,
                clean_text=clean_message,
                reason="question-like statement filtered",
            )

        if message_lower.endswith("?"):
            return self._result(True, message, msg_type, "question mark", clean_text=clean_message)

        if message_lower.endswith("..."):
            return self._result(False, message, msg_type, clean_text=clean_message, reason="trailing ellipsis")

        question_starters = (
            "is there ",
            "are there ",
            "can i ",
            "do you ",
            "did you ",
            "have you ",
            "does he ",
            "what's ",
            "who's ",
            "where's ",
            "when's ",
            "why's ",
            "how's ",
            "is it ",
            "are they ",
            "could i ",
            "i wonder if ",
            "anyone know ",
            "can anyone ",
            "does anyone ",
            "would anyone ",
            "should i ",
            "am i ",
        )
        matched_starter = next(
            (starter.strip() for starter in question_starters if self._has_phrase(message_lower, starter.strip())),
            None,
        )
        if matched_starter:
            return self._result(
                True,
                message,
                msg_type,
                "starter phrase",
                detail=matched_starter,
                clean_text=clean_message,
            )

        if not self.config.use_classifier:
            return self._result(False, message, msg_type, clean_text=clean_message, reason="no heuristic match")

        result = self._classify(clean_message)
        top_label = result["labels"][0]
        score = float(result["scores"][0])
        is_question = top_label == "information request" and score >= self.config.classifier_threshold

        return self._result(
            is_question,
            message,
            msg_type,
            "AI classifier" if is_question else None,
            score=score,
            detail=top_label,
            clean_text=clean_message,
            reason=None if is_question else "classifier below threshold or non-question label",
        )

    def _classify(self, text: str) -> Dict[str, Any]:
        if self._classifier is None:
            try:
                import torch
                from transformers import pipeline
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                except ImportError:
                    pass
            except ImportError as exc:
                raise RuntimeError(
                    "Classifier requested, but torch/transformers are not installed. "
                    "Run with --no-classifier or install project requirements."
                ) from exc

            device = 0 if torch.cuda.is_available() else -1
            dtype = torch.float16 if device == 0 else torch.float32
            self._classifier = pipeline(
                "zero-shot-classification",
                model="valhalla/distilbart-mnli-12-3",
                device=device,
                dtype=dtype,
                token=os.getenv("HF_TOKEN"),
            )

        labels = ["information request", "reaction or exclamation", "statement"]
        return self._classifier(text, candidate_labels=labels)

    def _result(
        self,
        is_question: bool,
        text: str,
        msg_type: str,
        method: Optional[str] = None,
        score: Optional[float] = None,
        detail: Optional[str] = None,
        clean_text: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> DetectionResult:
        return DetectionResult(
            index=-1,
            is_question=is_question,
            text=text,
            msg_type=msg_type,
            method=method,
            score=score,
            detail=detail,
            clean_text=clean_text,
            reason=reason,
        )

    @staticmethod
    def _has_phrase(message_lower: str, phrase: str) -> bool:
        return bool(re.search(r"(?<![a-z0-9]){}\b".format(re.escape(phrase)), message_lower))


def load_records(
    path: str,
    default_type: str = "chat",
    text_column: Optional[str] = None,
    user_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
    type_column: Optional[str] = None,
    records_path: Optional[str] = None,
) -> List[MessageRecord]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        raw_records = _load_csv(path)
    else:
        raw_records = _load_json_like(path, records_path)

    records = []
    for index, raw in enumerate(raw_records):
        text = _first_value(raw, [text_column] if text_column else TEXT_FIELDS)
        if text is None:
            continue

        msg_type = _first_value(raw, [type_column] if type_column else TYPE_FIELDS) or default_type
        user = _first_value(raw, [user_column] if user_column else USER_FIELDS)
        timestamp = _coerce_float(_first_value(raw, [timestamp_column] if timestamp_column else TIMESTAMP_FIELDS))

        if str(msg_type).lower() not in ("chat", "transcript"):
            if type_column:
                msg_type = str(msg_type).lower()
            else:
                msg_type = default_type

        records.append(
            MessageRecord(
                index=index,
                text=str(text),
                msg_type=str(msg_type).lower(),
                user=None if user is None else str(user),
                timestamp=timestamp,
                raw=raw,
            )
        )

    return records


def run_pipeline(records: Sequence[MessageRecord], detector: QuestionDetector) -> List[DetectionResult]:
    results = []
    for record in records:
        result = detector.detect(record.text, record.msg_type)
        result.index = record.index
        result.user = record.user
        result.timestamp = record.timestamp
        results.append(result)
    return results


def write_results(path: Optional[str], results: Sequence[DetectionResult], output_format: str) -> None:
    if output_format == "auto":
        output_format = _guess_output_format(path)

    rows = [asdict(result) for result in results]
    target = open(path, "w", encoding="utf-8", newline="") if path else sys.stdout
    try:
        if output_format == "csv":
            fieldnames = list(DetectionResult.__dataclass_fields__.keys())
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        elif output_format == "json":
            json.dump(rows, target, ensure_ascii=False, indent=2)
            target.write("\n")
        else:
            for row in rows:
                json.dump(row, target, ensure_ascii=False)
                target.write("\n")
    finally:
        if path:
            target.close()


def _load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_json_like(path: str, records_path: Optional[str]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    if not content:
        return []

    if _looks_like_jsonl(content):
        records = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
        return records

    data = json.loads(content)
    if records_path:
        data = _get_path(data, records_path)

    return _records_from_json(data)


def _records_from_json(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if not isinstance(data, dict):
        return []

    for key in KNOWN_CONTAINERS:
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    if _first_value(data, TEXT_FIELDS) is not None:
        return [data]

    nested = []
    for value in data.values():
        if isinstance(value, list):
            nested.extend(item for item in value if isinstance(item, dict) and _first_value(item, TEXT_FIELDS) is not None)
    return nested


def _looks_like_jsonl(content: str) -> bool:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return len(lines) > 1 and all(line.startswith("{") and line.endswith("}") for line in lines[:5])


def _first_value(data: Dict[str, Any], paths: Iterable[Optional[str]]) -> Any:
    for path in paths:
        if not path:
            continue
        value = _get_path(data, path)
        if value not in (None, ""):
            return value
    return None


def _get_path(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _guess_output_format(path: Optional[str]) -> str:
    if not path:
        return "jsonl"
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return "csv"
    if ext == ".json":
        return "json"
    return "jsonl"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run question detection on JSON, JSONL, or CSV datasets.")
    parser.add_argument("input", help="Input dataset path.")
    parser.add_argument("--output", "-o", help="Output path. Defaults to stdout.")
    parser.add_argument("--output-format", choices=("auto", "jsonl", "json", "csv"), default="auto")
    parser.add_argument("--records-path", help="Dot path to the JSON array containing records, if auto-detection is wrong.")
    parser.add_argument("--text-column", help="Text field/column name or dot path.")
    parser.add_argument("--user-column", help="User field/column name or dot path.")
    parser.add_argument("--timestamp-column", help="Timestamp field/column name or dot path.")
    parser.add_argument("--type-column", help="Message type field/column name or dot path.")
    parser.add_argument("--type", choices=("chat", "transcript"), default="chat", help="Default message type.")
    parser.add_argument("--include-non-questions", action="store_true", help="Write every row, not only detections.")
    parser.add_argument("--classifier", dest="use_classifier", action="store_true", default=True)
    parser.add_argument("--no-classifier", dest="use_classifier", action="store_false")
    parser.add_argument("--filter-short", dest="filter_short", action="store_true", default=False)
    parser.add_argument("--no-filter-short", dest="filter_short", action="store_false")
    parser.add_argument("--short-threshold", type=int, default=2)
    parser.add_argument("--classifier-threshold", type=float, default=0.60)
    parser.add_argument("--long-chat-word-limit", type=int, default=15)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = DetectionConfig(
        use_classifier=args.use_classifier,
        filter_short_questions=args.filter_short,
        short_question_threshold=args.short_threshold,
        classifier_threshold=args.classifier_threshold,
        long_chat_word_limit=args.long_chat_word_limit,
    )
    detector = QuestionDetector(config)

    records = load_records(
        args.input,
        default_type=args.type,
        text_column=args.text_column,
        user_column=args.user_column,
        timestamp_column=args.timestamp_column,
        type_column=args.type_column,
        records_path=args.records_path,
    )
    results = run_pipeline(records, detector)
    if not args.include_non_questions:
        results = [result for result in results if result.is_question]

    write_results(args.output, results, args.output_format)
    print(
        f"Processed {len(records)} records; wrote {len(results)} result(s).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
