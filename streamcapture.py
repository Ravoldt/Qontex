import subprocess
import time
import numpy as np
import os
import sys
import importlib
import gc
from pathlib import Path
import cv2
import threading
from collections import deque
import torch
from faster_whisper import WhisperModel
from hls_program_clock import HLSProgramClock
from utils import Message, create_stream_folder, get_config_value, get_streamer_name, is_likely_question, load_config, log_json, log_message, log_start_stop, shared_deque
import datetime

# Exported so it can be set by main.py
STREAM_START_TIME = None

current_video_timestamp = 0.0
chat_processing_timestamp = None

# In-memory buffer for video frames. Capture stores one frame per second, matching SharedDeque's 3-minute window.
video_frames = deque(maxlen=shared_deque.max_age)

# In-memory buffer for audio chunks (32ms each). Matches 3-minute window.
audio_chunks = deque(maxlen=int(shared_deque.max_age / 0.032))

_whisper_model = None
_vad_model = None
_mega_asr_module = None
_mega_asr_import_failed = False

DEFAULT_TRANSCRIPTION_MODEL_SETTINGS = {
    "use_silero_vad": False,
    "silero_max_seconds": 8.0,
    "memory_flush_seconds": 30.0,
}

TRANSCRIPTION_MODEL_SETTINGS = {
    # Future transcription backends can opt into Silero segmentation by setting this flag.
    "faster-whisper": {
        "use_silero_vad": True,
        "silero_max_seconds": 5.0,
    },
    "mega-asr": {
        "use_silero_vad": True,
        "silero_max_seconds": 8.0,
    },
}

def _transcription_model_settings(transcription_model):
    settings = DEFAULT_TRANSCRIPTION_MODEL_SETTINGS.copy()
    settings.update(TRANSCRIPTION_MODEL_SETTINGS.get(transcription_model, {}))
    return settings

class _MegaASRAdapter:
    def __init__(self, mega_asr_cls, repo_root=None):
        self._mega_asr_cls = mega_asr_cls
        self._repo_root = Path(repo_root).resolve() if repo_root else None
        self._model = None

    def _checkpoint_dir(self):
        configured = os.environ.get("MEGA_ASR_CKPT_DIR")
        if configured:
            return Path(configured).expanduser().resolve()
        if self._repo_root is not None:
            return self._repo_root / "ckpt" / "Mega-ASR"
        return Path("ckpt") / "Mega-ASR"

    def preload_model(self):
        if self._model is not None:
            return self._model

        ckpt_dir = self._checkpoint_dir()
        required_paths = [
            ckpt_dir / "Qwen3-ASR-1.7B",
            ckpt_dir / "mega-asr-merged",
            ckpt_dir / "audio_quality_router" / "best_acc_model.safetensors",
        ]
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "Mega-ASR weights are missing. Run "
                "`.\\.venv\\Scripts\\python.exe Mega-ASR\\scripts\\download.py` "
                f"or set MEGA_ASR_CKPT_DIR. Missing: {', '.join(missing_paths)}"
            )

        kwargs = {
            "model_path": ckpt_dir / "Qwen3-ASR-1.7B",
            "lora_dir": ckpt_dir / "mega-asr-merged",
            "router_checkpoint": ckpt_dir / "audio_quality_router" / "best_acc_model.safetensors",
            "backend": "transformers",
        }

        device_map = os.environ.get("MEGA_ASR_DEVICE_MAP")
        if device_map:
            kwargs["device_map"] = device_map

        routing = os.environ.get("MEGA_ASR_ROUTING")
        if routing is not None:
            kwargs["routing_enabled"] = routing.strip().lower() in ("1", "true", "yes", "y", "on")

        threshold = os.environ.get("MEGA_ASR_THRESHOLD")
        if threshold:
            kwargs["quality_threshold"] = float(threshold)

        try:
            self._model = self._mega_asr_cls(**kwargs)
            self._configure_generation_padding(self._model)
        except ModuleNotFoundError as e:
            raise RuntimeError(_mega_asr_install_hint(e)) from e
        return self._model

    def _configure_generation_padding(self, model):
        pad_token_id = self._resolve_pad_token_id(model)
        if pad_token_id is None:
            return

        for target in self._generation_config_targets(model):
            config = getattr(target, "config", None)
            if config is not None and getattr(config, "pad_token_id", None) is None:
                config.pad_token_id = pad_token_id

            generation_config = getattr(target, "generation_config", None)
            if generation_config is not None and getattr(generation_config, "pad_token_id", None) is None:
                generation_config.pad_token_id = pad_token_id

            if getattr(target, "pad_token_id", None) in (None, -1):
                try:
                    target.pad_token_id = pad_token_id
                except Exception:
                    pass

    def _resolve_pad_token_id(self, model):
        for tokenizer in self._tokenizer_candidates(model):
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is not None:
                return pad_token_id

        for target in self._generation_config_targets(model):
            generation_config = getattr(target, "generation_config", None)
            pad_token_id = getattr(generation_config, "pad_token_id", None)
            if pad_token_id is not None:
                return pad_token_id

        return None

    def _tokenizer_candidates(self, model):
        qwen_asr = getattr(model, "asr", None)
        qwen_wrapper = getattr(qwen_asr, "model", None)
        processors = (
            getattr(qwen_wrapper, "processor", None),
            getattr(qwen_asr, "processor", None),
            getattr(model, "processor", None),
        )

        for processor in processors:
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                yield tokenizer

    def _generation_config_targets(self, model):
        qwen_asr = getattr(model, "asr", None)
        qwen_wrapper = getattr(qwen_asr, "model", None)
        hf_model = getattr(qwen_wrapper, "model", None)
        thinker = getattr(hf_model, "thinker", None)

        seen = set()
        for target in (qwen_wrapper, hf_model, thinker):
            if target is not None and id(target) not in seen:
                seen.add(id(target))
                yield target

    def infer_from_memory(self, audio_array, sampling_rate=16000):
        model = self.preload_model()
        return model.infer_from_memory(
            audio_array,
            sampling_rate=sampling_rate,
            return_route=True,
        )

    def unload_model(self):
        if self._model is None:
            return
        self._model = None

def _resolve_mega_asr_paths(mega_asr_path):
    candidates = []
    if mega_asr_path:
        candidates.append(Path(mega_asr_path).expanduser())

    bundled_repo = Path(__file__).resolve().parent / "Mega-ASR"
    if bundled_repo.exists():
        candidates.append(bundled_repo)

    for candidate in candidates:
        try:
            path = candidate.resolve()
        except OSError:
            path = candidate

        if path.is_file():
            for parent in path.parents:
                src_dir = parent / "src"
                if (src_dir / "MegaASR" / "model" / "megaASR.py").is_file():
                    return src_dir, parent
            continue

        if not path.is_dir():
            continue

        if (path / "src" / "MegaASR" / "model" / "megaASR.py").is_file():
            return path / "src", path
        if (path / "MegaASR" / "model" / "megaASR.py").is_file():
            return path, path.parent
        if path.name == "MegaASR" and (path / "model" / "megaASR.py").is_file():
            return path.parent, path.parent.parent
        if path.name == "model" and (path / "megaASR.py").is_file():
            src_dir = path.parent.parent
            return src_dir, src_dir.parent

    return None, None

def _mega_asr_install_hint(error):
    if isinstance(error, ModuleNotFoundError):
        return (
            f"{error}. Install Mega-ASR's dependencies with "
            "`.\\.venv\\Scripts\\python.exe -m pip install qwen-asr soundfile scipy`."
        )
    return str(error)

def _import_mega_asr():
    global _mega_asr_module, _mega_asr_import_failed
    
    if _mega_asr_module is not None:
        return _mega_asr_module
    if _mega_asr_import_failed:
        return None
        
    mega_asr_path = get_config_value("MEGA_ASR_PATH")
            
    try:
        src_dir, repo_root = _resolve_mega_asr_paths(mega_asr_path)
        if mega_asr_path and src_dir is None:
            raise ImportError(
                f"Could not find MegaASR.model.megaASR from MEGA_ASR_PATH={mega_asr_path!r}. "
                "Set MEGA_ASR_PATH to the Mega-ASR repo root or its src folder."
            )

        if src_dir is not None and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        module = importlib.import_module("MegaASR.model.megaASR")
        _mega_asr_module = _MegaASRAdapter(module.MegaASR, repo_root)
        return _mega_asr_module
    except Exception as e:
        _mega_asr_import_failed = True
        print(f"\n[QONTEX] ERROR: Failed to import Mega-ASR. Details: {_mega_asr_install_hint(e)}\n")
        return None

def _mega_asr_segments(result):
    if result is None:
        return []
    if isinstance(result, str):
        return [(result, 0.0)]
    if isinstance(result, dict):
        text = (
            result.get("text")
            or result.get("transcription")
            or result.get("result")
            or result.get("output")
            or ""
        )
        if isinstance(text, (dict, list, tuple)):
            return _mega_asr_segments(text)
        return [(str(text), float(result.get("start", 0.0) or 0.0))] if text else []
    if isinstance(result, (list, tuple)):
        segments = []
        for item in result:
            if hasattr(item, "text"):
                segments.append((item.text, float(getattr(item, "start", 0.0) or 0.0)))
            elif isinstance(item, dict):
                segments.extend(_mega_asr_segments(item))
            elif isinstance(item, str):
                segments.append((item, 0.0))
        return segments
    if hasattr(result, "text"):
        return [(result.text, float(getattr(result, "start", 0.0) or 0.0))]
    return []

def _load_silero_vad(device):
    global _vad_model

    if _vad_model is None:
        print(f"[QONTEX] Loading Silero VAD model into {'VRAM' if device == 'cuda' else 'RAM'}...")
        _vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
        _vad_model = _vad_model.to(device)
        _vad_model.eval()

    return _vad_model

def preload_models(transcription_model="faster-whisper", use_silero_vad=None):
    global _whisper_model

    settings = _transcription_model_settings(transcription_model)
    if use_silero_vad is None:
        use_silero_vad = settings["use_silero_vad"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if transcription_model == "faster-whisper":
        if device == "cpu":
            print("[QONTEX] WARNING: CUDA is not available. Loading models on CPU (this will be slow).")
            compute_type = "int8"
        else:
            compute_type = "float16"

        if _whisper_model is None:
            print(f"[QONTEX] Loading faster-whisper model into {'VRAM' if device == 'cuda' else 'RAM'}... (this may take a moment)")
            _whisper_model = WhisperModel("large-v3", device=device, compute_type=compute_type) 

    elif transcription_model == "mega-asr":
        megaASR = _import_mega_asr()
        if not megaASR:
            raise RuntimeError("Mega-ASR could not be loaded; see the error above.")
        print("[QONTEX] Preloading Mega-ASR model...")
        try:
            megaASR.preload_model()
        except Exception as e:
            print(f"\n[QONTEX] ERROR: Failed to preload Mega-ASR. Details: {e}\n")
            raise

    if use_silero_vad:
        _load_silero_vad(device)

def unload_models():
    """Release transcription/VAD models and return GPU memory to PyTorch."""
    global _whisper_model, _vad_model

    _whisper_model = None
    _vad_model = None

    if _mega_asr_module is not None:
        _mega_asr_module.unload_model()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _hls_live_edge_segments():
    try:
        return max(1, int(get_config_value("HLS_LIVE_EDGE_SEGMENTS", 3)))
    except (TypeError, ValueError):
        return 3


def _is_hls_playlist_url(url):
    return ".m3u8" in (url or "").split("?", 1)[0].lower()


def _select_stream(streams, quality):
    if quality == "audio_only":
        candidates = ("audio_only", "audio", "best")
    elif quality == "best":
        candidates = ("best", "source", "chunked")
    else:
        candidates = (quality, "best")

    for candidate in candidates:
        stream = streams.get(candidate)
        if stream is not None:
            return stream, candidate

    available = ", ".join(sorted(streams.keys())) or "none"
    raise RuntimeError(f"Requested stream quality '{quality}' was not available. Available: {available}")


def _stream_to_url(stream):
    try:
        return stream.to_url()
    except TypeError:
        return stream.to_manifest_url()


def _resolve_stream_url(source, quality, live_edge_segments):
    from streamlink import Streamlink
    from streamlink.options import Options

    session = Streamlink()
    session.set_option("hls-live-edge", live_edge_segments)

    streams = session.streams(
        source,
        options=Options({"low-latency": False}),
    )
    if not streams:
        raise RuntimeError("No streams were returned. Is the streamer offline?")

    stream, selected_quality = _select_stream(streams, quality)
    substreams = list(getattr(stream, "substreams", None) or [])
    if substreams:
        video_url = _stream_to_url(substreams[0])
        audio_url = _stream_to_url(substreams[1] if len(substreams) > 1 else substreams[0])
    else:
        audio_url = _stream_to_url(stream)
        video_url = audio_url

    if selected_quality != quality:
        print(f"[QONTEX] Requested {quality}; using {selected_quality}.")
    return audio_url, session, video_url


def _resolve_stream_url_with_cli(source, quality):
    return subprocess.check_output(
        ["streamlink", "--stream-url", source, quality],
        stderr=subprocess.STDOUT,
    ).decode("utf-8").strip()


def start_audio_capture(source, process_fast=False, enable_video=None, stop_event=None):
    """Starts a background process to extract audio from a video or livestream."""
    if not source.startswith("http") and "." not in source:
        source = f"https://www.twitch.tv/{source}"

    is_livestream = "twitch.tv" in source or source.startswith("http")
    m3u8_url = None
    video_m3u8_url = None
    hls_clock = None
    streamlink_session = None
    
    command = ["ffmpeg"]
    
    if not is_livestream and not process_fast:
        command.append("-re")
        
    if is_livestream:
        print(f"[QONTEX] Resolving stream URL for {source}...")
        try:
            if enable_video is None:
                enable_video = get_config_value("ENABLE_VISUAL_CONTEXT", False)
            quality = "best" if enable_video else "audio_only"
            live_edge_segments = _hls_live_edge_segments()
            try:
                m3u8_url, streamlink_session, video_m3u8_url = _resolve_stream_url(source, quality, live_edge_segments)
            except ModuleNotFoundError:
                m3u8_url = _resolve_stream_url_with_cli(source, quality)
                video_m3u8_url = m3u8_url

            input_url = m3u8_url
            if _is_hls_playlist_url(m3u8_url):
                hls_clock = HLSProgramClock(
                    m3u8_url,
                    session=streamlink_session,
                    live_edge_segments=live_edge_segments,
                )
                controlled_playlist = hls_clock.prime_controlled_playlist()
                if controlled_playlist:
                    print(f"[QONTEX] HLS program clock aligned ({hls_clock.debug_description(STREAM_START_TIME)}).")
                    hls_clock.start(stop_event)
                    input_url = controlled_playlist
                    command.extend([
                        "-protocol_whitelist", "file,http,https,tcp,tls,crypto",
                        "-live_start_index", "0",
                    ])
                else:
                    print("[QONTEX] HLS program timestamps were unavailable; using local capture timing.")
                    command.extend(["-live_start_index", f"-{live_edge_segments}"])

            command.extend(["-i", input_url])
        except subprocess.CalledProcessError as e:
            print(f"[QONTEX] Error resolving stream. Is the streamer offline?\nDetails: {e.output.decode('utf-8').strip()}")
            return None, None, None
        except FileNotFoundError:
            print("[QONTEX] Error: Streamlink or ffmpeg is not installed or not in your system PATH.")
            return None, None, None
        except Exception as e:
            print(f"[QONTEX] Error resolving stream: {e}")
            return None, None, None
    else:
        command.extend(["-i", source])
        
    command.extend([
        "-vn", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"
    ])
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        return process, video_m3u8_url or m3u8_url, hls_clock
        
    except FileNotFoundError:
        print("[QONTEX] Error: ffmpeg is not installed or not in your system PATH.")
        return None, None, None

def capture_video_frames(source_url, stop_event=None, target_fps=1.0, state=None):
    """Background thread to capture a specific number of frames per second from the video stream."""
    global video_frames
    if state:
        frame_buffer = state.configure_video_buffer(target_fps)
    else:
        # Resize the deque to match the target FPS while keeping the 3-minute time window
        video_frames = deque(maxlen=int(shared_deque.max_age * target_fps))
        frame_buffer = video_frames
    cap = cv2.VideoCapture(source_url)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
        
    frame_interval = max(1, int(fps / target_fps))
    frame_count = 0
    
    while not (stop_event and stop_event.is_set()):
        # Grab advances the stream without doing heavy pixel decoding
        ret = cap.grab()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            ret, frame = cap.retrieve() # Only decode the frame we are actually keeping
            if ret:
                frame_buffer.append(frame)
            
        frame_count += 1
        
    cap.release()

def run_capture_loop(source, log_folder, process_fast=False, question_handler=None, stop_event=None, transcript_user=None, target_fps=1.0, state=None):
    """Main capture loop running as a background thread."""
    global STREAM_START_TIME, _whisper_model, _vad_model

    if state:
        log_folder = state.log_folder
        process_fast = state.config.process_fast
        target_fps = state.config.visual_context_fps

    if transcript_user is None:
        transcript_user = state.config.stream_name if state else get_streamer_name(source)
    
    transcription_model_name = state.config.transcription_model if state else get_config_value("TRANSCRIPTION_MODEL", "faster-whisper")
    is_livestream = "twitch.tv" in source or source.startswith("http") or (not os.path.isfile(source) and "." not in source)
    model_settings = _transcription_model_settings(transcription_model_name)
    uses_faster_whisper = transcription_model_name == "faster-whisper"
    uses_mega_asr = transcription_model_name == "mega-asr"
    use_silero_vad = bool(model_settings["use_silero_vad"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if uses_faster_whisper and (_whisper_model is None or (use_silero_vad and _vad_model is None)):
        preload_models(transcription_model_name, use_silero_vad=use_silero_vad)
    elif uses_mega_asr:
        preload_models(transcription_model_name, use_silero_vad=use_silero_vad)
    elif use_silero_vad and _vad_model is None:
        _load_silero_vad(device)
    
    model = _whisper_model if uses_faster_whisper else None
    vad_model = _vad_model if use_silero_vad else None
    megaASR = _import_mega_asr() if uses_mega_asr else None
    
    enable_video = state.config.enable_visual_context if state else get_config_value("ENABLE_VISUAL_CONTEXT", False)
    audio_process, m3u8_url, hls_clock = start_audio_capture(
        source,
        process_fast=process_fast,
        enable_video=enable_video,
        stop_event=stop_event,
    )
    if state:
        state.capture_process = audio_process

    if audio_process:
        stream_ended = False
        if not process_fast and enable_video:
            target_url = m3u8_url if m3u8_url else source
            video_thread = threading.Thread(target=capture_video_frames, args=(target_url, stop_event, target_fps, state), daemon=True)
            video_thread.start()
        
        loop_name = "Silero VAD loop" if use_silero_vad else "memory capture loop"
        print(f"[QONTEX] Audio stream captured! Running {loop_name}...")
        
        try:                      
            # Keep 32ms reads for timing, VAD, and audio context storage.
            CHUNK_DURATION = 0.032  
            CHUNK_SIZE = int(16000 * 2 * CHUNK_DURATION) 
            
            SPEECH_THRESHOLD = 0.5     # AI Confidence: 50% probability it is human speech
            MAX_SILENCE_CHUNKS = 30    # ~1.0 second of silence triggers the end of a sentence
            VAD_MAX_CHUNKS = max(1, int(float(model_settings["silero_max_seconds"]) / CHUNK_DURATION))
            MEMORY_FLUSH_CHUNKS = max(1, int(float(model_settings["memory_flush_seconds"]) / CHUNK_DURATION))
            
            audio_buffer = []
            silence_counter = 0
            is_speaking = False
            
            total_chunks = 0
            audio_start_time = None

            def hls_timestamp_for_offset(audio_offset):
                if process_fast or hls_clock is None:
                    return None
                return hls_clock.timestamp_for_audio_offset(audio_offset, STREAM_START_TIME)

            def process_transcript_segment(text, start_offset, chunk_base_time, audio_start_time_ref):
                if not text or not text.strip():
                    return
                audio_offset = chunk_base_time + start_offset
                hls_msg_time = hls_timestamp_for_offset(audio_offset)
                if hls_msg_time is not None:
                    msg_time = hls_msg_time
                elif STREAM_START_TIME is None or process_fast:
                    msg_time = chunk_base_time + start_offset
                else:
                    segment_wall_time = audio_start_time_ref + chunk_base_time + start_offset
                    msg_time = segment_wall_time - STREAM_START_TIME

                msg_time = max(0.0, msg_time)

                msg = Message(msg_time, "transcript", text.strip(), user=transcript_user)
                print(msg)
                log_message(log_folder, "transcript.log", msg)
                log_json(log_folder, "merged.json", msg.to_dict())
                if state:
                    state.add_message(msg)
                else:
                    shared_deque.add_message(msg)
                should_detect_question = question_handler or get_config_value("LOG_QUESTION_DETECTIONS", True)
                if should_detect_question and is_likely_question(msg.text, msg.type) and question_handler:
                    threading.Thread(target=question_handler, args=(msg,), daemon=True).start()

            def transcribe_buffered_audio(full_audio, chunk_base_time):
                if uses_faster_whisper:
                    segments, _ = model.transcribe(full_audio, beam_size=5, vad_filter=False)
                    for segment in segments:
                        process_transcript_segment(segment.text, segment.start, chunk_base_time, audio_start_time)
                elif uses_mega_asr:
                    if not megaASR:
                        return
                    result = megaASR.infer_from_memory(full_audio)
                    segments = _mega_asr_segments(result)
                    if segments:
                        for text, start in segments:
                            process_transcript_segment(text, start, chunk_base_time, audio_start_time)
            
            while not (stop_event and stop_event.is_set()):
                in_bytes = audio_process.stdout.read(CHUNK_SIZE)
                if not in_bytes or len(in_bytes) < CHUNK_SIZE:
                    stream_ended = True
                    break 

                if state:
                    state.audio_chunks.append(in_bytes)
                else:
                    audio_chunks.append(in_bytes)

                if audio_start_time is None:
                    audio_start_time = time.time()
                    
                total_chunks += 1

                global current_video_timestamp
                audio_elapsed = total_chunks * CHUNK_DURATION
                hls_current_time = hls_timestamp_for_offset(audio_elapsed)
                if hls_current_time is not None:
                    current_video_timestamp = hls_current_time
                elif STREAM_START_TIME is None or process_fast:
                    current_video_timestamp = total_chunks * CHUNK_DURATION
                else:
                    current_video_timestamp = max(0.0, time.time() - STREAM_START_TIME)
                if state:
                    state.current_video_timestamp = current_video_timestamp

                global chat_processing_timestamp
                wait_for_chat_ts = state.chat_processing_timestamp if state else chat_processing_timestamp
                if wait_for_chat_ts is not None:
                    while current_video_timestamp > wait_for_chat_ts + 1.0 and not (stop_event and stop_event.is_set()):
                        time.sleep(0.01)
                        wait_for_chat_ts = state.chat_processing_timestamp if state else chat_processing_timestamp

                audio_data = np.frombuffer(in_bytes, np.int16).astype(np.float32) / 32768.0
                
                if use_silero_vad:
                    audio_tensor = torch.from_numpy(audio_data).to(device)
                    speech_prob = vad_model(audio_tensor, 16000).item()
                    
                    if speech_prob > SPEECH_THRESHOLD:
                        is_speaking = True
                        silence_counter = 0
                        audio_buffer.append(audio_data)
                    elif is_speaking:
                        silence_counter += 1
                        audio_buffer.append(audio_data)

                    if audio_buffer and (
                        silence_counter >= MAX_SILENCE_CHUNKS
                        or len(audio_buffer) >= VAD_MAX_CHUNKS
                    ):
                        full_audio = np.concatenate(audio_buffer)
                        chunk_base_time = (total_chunks - len(audio_buffer)) * CHUNK_DURATION
                        transcribe_buffered_audio(full_audio, chunk_base_time)
                        audio_buffer = []
                        is_speaking = False
                        silence_counter = 0

                else:
                    audio_buffer.append(audio_data)
                    if not process_fast and len(audio_buffer) >= MEMORY_FLUSH_CHUNKS:
                        full_audio = np.concatenate(audio_buffer)
                        chunk_base_time = (total_chunks - len(audio_buffer)) * CHUNK_DURATION
                        transcribe_buffered_audio(full_audio, chunk_base_time)
                        audio_buffer = []

            # Flush and transcribe any remaining audio when the stream ends
            if len(audio_buffer) > 0:
                full_audio = np.concatenate(audio_buffer)
                chunk_base_time = (total_chunks - len(audio_buffer)) * CHUNK_DURATION
                transcribe_buffered_audio(full_audio, chunk_base_time)
                
        except Exception as e:
            print(f"[QONTEX] Error in capture loop: {e}")
        finally:
            if state and state.capture_process is audio_process:
                state.capture_process = None
            if audio_process.poll() is None:
                audio_process.terminate()
            audio_process.wait()
            if hls_clock is not None:
                hls_clock.cleanup()

        if is_livestream and stream_ended and not (stop_event and stop_event.is_set()):
            return "stream_ended"
        return "stopped"

    if state:
        state.capture_process = None
    return "unavailable" if is_livestream else "stopped"

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    STREAM_START_TIME = time.time()
    try:
        config = load_config()
    except FileNotFoundError:
        print("[QONTEX] CRITICAL ERROR: config.toml is missing!")
        exit(1)
        
    source = config.get("CHANNEL", "")
    is_local_video = os.path.isfile(source)
    
    if not is_local_video and not source.startswith("http"):
        source = source.lstrip('#')
    
    if not source:
        print("[QONTEX] Please set the CHANNEL in config.toml!")
        exit(1)
        
    process_fast = config.get("PROCESS_FAST", False) if is_local_video else False

    streamer_name = get_streamer_name(source)
    stream_start = datetime.datetime.now()
    log_folder = create_stream_folder(streamer_name, stream_start)
    log_start_stop(log_folder, "start")
    
    print(f"[QONTEX] Testing capture loop for {source}...")
    run_capture_loop(source, log_folder, process_fast=process_fast)
