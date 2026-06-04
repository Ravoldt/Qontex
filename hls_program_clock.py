import datetime
import os
import re
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin


_EXTINF_RE = re.compile(r"#EXTINF:([0-9]+(?:\.[0-9]+)?)")


@dataclass
class HLSTimedSegment:
    sequence: int
    duration: float
    uri: Optional[str] = None
    lines: tuple[str, ...] = ()
    program_wall_time: Optional[float] = None
    elapsed_seconds: Optional[float] = None

    @property
    def has_timing(self):
        return self.elapsed_seconds is not None or self.program_wall_time is not None


@dataclass
class HLSPlaylistSnapshot:
    media_sequence: int
    target_duration: float
    segments: list[HLSTimedSegment]
    loaded_at_wall: float


class HLSProgramClock:
    """Maps decoded audio offsets back to the HLS segment production timeline."""

    def __init__(self, playlist_url, session=None, live_edge_segments=3):
        self.playlist_url = playlist_url
        self.session = session
        self.live_edge_segments = max(1, int(live_edge_segments or 3))
        self._lock = threading.Lock()
        self._base_segment = None
        self._latest_snapshot = None
        self._thread = None
        self._stop_event = threading.Event()
        self._controlled_playlist_path = None
        self._controlled_segments = {}
        self._controlled_first_sequence = None

    def prime(self):
        snapshot = self.fetch_snapshot()
        if not snapshot:
            return False

        with self._lock:
            self._latest_snapshot = snapshot
            if self._base_segment is None:
                segment = self._select_start_segment(snapshot)
                if segment and segment.has_timing:
                    self._base_segment = segment
                    return True

        return self.is_ready()

    def prime_controlled_playlist(self):
        snapshot = self.fetch_snapshot()
        if not snapshot:
            return None

        with self._lock:
            self._latest_snapshot = snapshot
            segment = self._select_start_segment(snapshot)
            if not segment or not segment.has_timing:
                return None

            self._base_segment = segment
            self._controlled_first_sequence = segment.sequence
            self._merge_snapshot_segments(snapshot)
            self._ensure_controlled_playlist_path()
            self._write_controlled_playlist()
            return self._controlled_playlist_path

    def start(self, stop_event=None):
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            args=(stop_event,),
            daemon=True,
        )
        self._thread.start()

    def is_ready(self):
        with self._lock:
            return self._base_segment is not None

    def timestamp_for_audio_offset(self, offset_seconds, stream_start_wall=None):
        with self._lock:
            base_segment = self._base_segment

        if not base_segment:
            return None

        try:
            offset_seconds = float(offset_seconds)
        except (TypeError, ValueError):
            return None

        if base_segment.elapsed_seconds is not None:
            return max(0.0, base_segment.elapsed_seconds + offset_seconds)

        if base_segment.program_wall_time is None or stream_start_wall is None:
            return None

        return max(0.0, (base_segment.program_wall_time - float(stream_start_wall)) + offset_seconds)

    def debug_description(self, stream_start_wall=None):
        with self._lock:
            segment = self._base_segment

        if not segment:
            return "not aligned"

        timestamp = self.timestamp_for_audio_offset(0.0, stream_start_wall)
        if timestamp is None:
            return f"sequence {segment.sequence}"
        return f"sequence {segment.sequence} at {timestamp:.3f}s"

    def fetch_snapshot(self):
        try:
            text, base_uri = self._fetch_playlist_text()
            return parse_hls_timing_playlist(text, base_uri=base_uri)
        except Exception:
            return None

    def _poll_loop(self, stop_event):
        while not self._should_stop(stop_event):
            snapshot = self.fetch_snapshot()
            if snapshot:
                with self._lock:
                    self._latest_snapshot = snapshot
                    if self._controlled_playlist_path:
                        self._merge_snapshot_segments(snapshot)
                        self._write_controlled_playlist()
                    if self._base_segment is None:
                        segment = self._select_start_segment(snapshot)
                        if segment and segment.has_timing:
                            self._base_segment = segment

                sleep_seconds = snapshot.target_duration or 2.0
            else:
                sleep_seconds = 2.0

            end_time = time.time() + max(0.5, min(float(sleep_seconds), 5.0))
            while time.time() < end_time:
                if self._should_stop(stop_event):
                    return
                time.sleep(0.1)

    def _should_stop(self, stop_event):
        return self._stop_event.is_set() or bool(stop_event and stop_event.is_set())

    def _fetch_playlist_text(self):
        if self.session is not None and hasattr(self.session, "http"):
            response = self.session.http.get(self.playlist_url)
            response.encoding = "utf-8"
            return response.text, response.url

        request = urllib.request.Request(
            self.playlist_url,
            headers={"User-Agent": "Qontex/1.0"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read().decode("utf-8", errors="replace"), response.geturl()

    def _select_start_segment(self, snapshot):
        if not snapshot.segments:
            return None

        edge = min(self.live_edge_segments, len(snapshot.segments))
        segment = snapshot.segments[-edge]
        return segment if segment.has_timing else None

    def _merge_snapshot_segments(self, snapshot):
        if self._controlled_first_sequence is None:
            return

        for segment in snapshot.segments:
            if segment.sequence >= self._controlled_first_sequence:
                self._controlled_segments[segment.sequence] = segment

    def _ensure_controlled_playlist_path(self):
        if self._controlled_playlist_path:
            return

        fd, path = tempfile.mkstemp(prefix="qontex_hls_", suffix=".m3u8")
        os.close(fd)
        self._controlled_playlist_path = path

    def _write_controlled_playlist(self):
        if not self._controlled_playlist_path or not self._controlled_segments:
            return

        sequences = sorted(self._controlled_segments)
        first_sequence = sequences[0]
        target_duration = self._latest_snapshot.target_duration if self._latest_snapshot else 0.0
        if not target_duration:
            target_duration = max(
                (segment.duration for segment in self._controlled_segments.values()),
                default=2.0,
            )

        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{max(1, int(target_duration + 0.999))}",
            f"#EXT-X-MEDIA-SEQUENCE:{first_sequence}",
            "#EXT-X-PLAYLIST-TYPE:EVENT",
        ]
        for sequence in sequences:
            segment = self._controlled_segments[sequence]
            lines.extend(segment.lines)

        self._write_text_atomic(self._controlled_playlist_path, "\n".join(lines) + "\n")

    @staticmethod
    def _write_text_atomic(path, text):
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

        try:
            os.replace(tmp_path, path)
        except PermissionError:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def cleanup(self):
        self._stop_event.set()
        with self._lock:
            path = self._controlled_playlist_path
            self._controlled_playlist_path = None

        if not path:
            return

        try:
            os.remove(path)
        except OSError:
            pass


def parse_hls_timing_playlist(text, base_uri=None):
    media_sequence = 0
    target_duration = 0.0
    segments = []

    pending_duration = None
    pending_program_wall = None
    pending_elapsed = None
    pending_lines = []
    active_key_line = None
    active_map_line = None
    inferred_program_wall = None
    inferred_elapsed = None
    twitch_total_seconds = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            media_sequence = _parse_int(line.split(":", 1)[1], media_sequence)
            continue

        if line.startswith("#EXT-X-TARGETDURATION:"):
            target_duration = _parse_float(line.split(":", 1)[1], target_duration)
            continue

        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            pending_program_wall = _parse_program_date_time(line.split(":", 1)[1])
            pending_lines.append(line)
            continue

        if line.startswith("#EXT-X-TWITCH-ELAPSED-SECS:"):
            pending_elapsed = _parse_float(line.split(":", 1)[1], None)
            pending_lines.append(line)
            continue

        if line.startswith("#EXT-X-TWITCH-TOTAL-SECS:"):
            twitch_total_seconds = _parse_float(line.split(":", 1)[1], None)
            continue

        if line.startswith("#EXT-X-DISCONTINUITY"):
            inferred_program_wall = None
            inferred_elapsed = None
            pending_lines.append(line)
            continue

        if line.startswith("#EXT-X-KEY:"):
            active_key_line = _rewrite_hls_tag_uri(line, base_uri)
            continue

        if line.startswith("#EXT-X-MAP:"):
            active_map_line = _rewrite_hls_tag_uri(line, base_uri)
            continue

        if line.startswith("#EXT-X-TWITCH-PREFETCH"):
            continue

        if line.startswith("#EXT-X-ENDLIST"):
            continue

        extinf_match = _EXTINF_RE.match(line)
        if extinf_match:
            pending_duration = _parse_float(extinf_match.group(1), 0.0)
            pending_lines.append(line)
            continue

        if line.startswith("#"):
            if pending_lines:
                pending_lines.append(_rewrite_hls_tag_uri(line, base_uri))
            continue

        if pending_duration is None:
            continue

        if pending_program_wall is not None:
            program_wall = pending_program_wall
        else:
            program_wall = inferred_program_wall

        if pending_elapsed is not None:
            elapsed = pending_elapsed
        else:
            elapsed = inferred_elapsed

        segment_uri = urljoin(base_uri or "", line)
        segment_lines = []
        if active_key_line:
            segment_lines.append(active_key_line)
        if active_map_line:
            segment_lines.append(active_map_line)
        segment_lines.extend(pending_lines)
        segment_lines.append(segment_uri)

        segment = HLSTimedSegment(
            sequence=media_sequence + len(segments),
            duration=pending_duration,
            uri=segment_uri,
            lines=tuple(segment_lines),
            program_wall_time=program_wall,
            elapsed_seconds=elapsed,
        )
        segments.append(segment)

        inferred_program_wall = (
            None if program_wall is None else program_wall + pending_duration
        )
        inferred_elapsed = None if elapsed is None else elapsed + pending_duration
        pending_duration = None
        pending_program_wall = None
        pending_elapsed = None
        pending_lines = []

    if twitch_total_seconds is not None and not any(
        segment.elapsed_seconds is not None for segment in segments
    ):
        elapsed = twitch_total_seconds - sum(segment.duration for segment in segments)
        for segment in segments:
            segment.elapsed_seconds = elapsed
            elapsed += segment.duration

    return HLSPlaylistSnapshot(
        media_sequence=media_sequence,
        target_duration=target_duration,
        segments=segments,
        loaded_at_wall=time.time(),
    )


def _parse_program_date_time(value):
    value = value.strip()
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"

    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)

    return parsed.timestamp()


def _parse_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rewrite_hls_tag_uri(line, base_uri):
    if not base_uri or "URI=" not in line:
        return line

    def replace(match):
        return f'{match.group(1)}"{urljoin(base_uri, match.group(2))}"'

    return re.sub(r'(URI=)"([^"]+)"', replace, line)
