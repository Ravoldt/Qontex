import subprocess
import time
import numpy as np
import os
import sys
import importlib
import tempfile
import wave
from pathlib import Path
import cv2
import threading
from collections import deque
import torch
from faster_whisper import WhisperModel
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
        except ModuleNotFoundError as e:
            raise RuntimeError(_mega_asr_install_hint(e)) from e
        return self._model

    def infer_from_memory(self, audio_array, sampling_rate=16000):
        model = self.preload_model()
        wav_path = _write_temp_wav(audio_array, sampling_rate)
        try:
            return model.infer(wav_path, return_route=True)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

def _write_temp_wav(audio_array, sampling_rate=16000):
    pcm = np.asarray(audio_array)
    if pcm.dtype != np.int16:
        pcm = np.clip(pcm, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)

    fd, wav_path = tempfile.mkstemp(prefix="qontex-mega-asr-", suffix=".wav")
    os.close(fd)
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(pcm.tobytes())
    return wav_path

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

def preload_models(transcription_model="faster-whisper"):
    global _whisper_model, _vad_model
    
    if transcription_model == "faster-whisper":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("[QONTEX] WARNING: CUDA is not available. Loading models on CPU (this will be slow).")
            compute_type = "int8"
        else:
            compute_type = "float16"

        if _whisper_model is None:
            print(f"[QONTEX] Loading faster-whisper model into {'VRAM' if device == 'cuda' else 'RAM'}... (this may take a moment)")
            _whisper_model = WhisperModel("large-v3", device=device, compute_type=compute_type) 
        
        if _vad_model is None:
            print(f"[QONTEX] Loading Silero VAD model into {'VRAM' if device == 'cuda' else 'RAM'}...")
            _vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
            _vad_model = _vad_model.to(device)
            _vad_model.eval()
            
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

def start_audio_capture(source, process_fast=False, enable_video=None):
    """Starts a background process to extract audio from a video or livestream."""
    if not source.startswith("http") and "." not in source:
        source = f"https://www.twitch.tv/{source}"

    is_livestream = "twitch.tv" in source or source.startswith("http")
    m3u8_url = None
    
    command = ["ffmpeg"]
    
    if not is_livestream and not process_fast:
        command.append("-re")
        
    if is_livestream:
        print(f"[QONTEX] Resolving stream URL for {source}...")
        try:
            if enable_video is None:
                enable_video = get_config_value("ENABLE_VISUAL_CONTEXT", False)
            quality = "best" if enable_video else "audio_only"
            m3u8_url = subprocess.check_output(
                ["streamlink", "--stream-url", source, quality],
                stderr=subprocess.STDOUT
            ).decode("utf-8").strip()
            command.extend(["-i", m3u8_url])
        except subprocess.CalledProcessError as e:
            print(f"[QONTEX] Error resolving stream. Is the streamer offline?\nDetails: {e.output.decode('utf-8').strip()}")
            return None, None
        except FileNotFoundError:
            print("[QONTEX] Error: Streamlink is not installed or not in your system PATH.")
            return None, None
        except Exception as e:
            print(f"[QONTEX] Error resolving stream: {e}")
            return None, None
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
        return process, m3u8_url
        
    except FileNotFoundError:
        print("[QONTEX] Error: Streamlink is not installed or not in your system PATH.")
        return None, None

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
    
    if (transcription_model_name == "faster-whisper" and (_whisper_model is None or _vad_model is None)) or transcription_model_name == "mega-asr":
        preload_models(transcription_model_name)
    
    model = _whisper_model
    vad_model = _vad_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    enable_video = state.config.enable_visual_context if state else get_config_value("ENABLE_VISUAL_CONTEXT", False)
    audio_process, m3u8_url = start_audio_capture(source, process_fast=process_fast, enable_video=enable_video)

    if audio_process:
        if not process_fast and enable_video:
            target_url = m3u8_url if m3u8_url else source
            video_thread = threading.Thread(target=capture_video_frames, args=(target_url, stop_event, target_fps, state), daemon=True)
            video_thread.start()
        
        print("[QONTEX] Audio stream captured! Running AI-VAD loop...")
        
        try:                      
            # Silero natively prefers exactly 32ms chunks (512 samples at 16kHz)
            CHUNK_DURATION = 0.032  
            CHUNK_SIZE = int(16000 * 2 * CHUNK_DURATION) 
            
            SPEECH_THRESHOLD = 0.5     # AI Confidence: 50% probability it is human speech
            MAX_SILENCE_CHUNKS = 30    # ~1.0 second of silence triggers the end of a sentence
            MAX_CHUNK_LIMIT = 156      # ~5.0 seconds maximum chunk length 
            
            audio_buffer = []
            silence_counter = 0
            is_speaking = False
            
            total_chunks = 0
            audio_start_time = None
            
            while not (stop_event and stop_event.is_set()):
                in_bytes = audio_process.stdout.read(CHUNK_SIZE)
                if not in_bytes or len(in_bytes) < CHUNK_SIZE:
                    break 

                if state:
                    state.audio_chunks.append(in_bytes)
                else:
                    audio_chunks.append(in_bytes)

                if audio_start_time is None:
                    audio_start_time = time.time()
                    
                total_chunks += 1

                def process_transcript_segment(text, start_offset, chunk_base_time, audio_start_time_ref):
                    if not text or not text.strip():
                        return
                    if STREAM_START_TIME is None or process_fast:
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


                global current_video_timestamp
                if STREAM_START_TIME is None or process_fast:
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
                
                if transcription_model_name == "faster-whisper":
                    audio_tensor = torch.from_numpy(audio_data).to(device)
                    speech_prob = vad_model(audio_tensor, 16000).item()
                    
                    if speech_prob > SPEECH_THRESHOLD:
                        is_speaking = True
                        silence_counter = 0
                        audio_buffer.append(audio_data)
                    elif is_speaking:
                        silence_counter += 1
                        audio_buffer.append(audio_data)
                        
                        if silence_counter >= MAX_SILENCE_CHUNKS or len(audio_buffer) >= MAX_CHUNK_LIMIT:
                            full_audio = np.concatenate(audio_buffer)
                            
                            segments, _ = model.transcribe(full_audio, beam_size=5, vad_filter=False)
                            chunk_base_time = (total_chunks - len(audio_buffer)) * CHUNK_DURATION
                            
                            for segment in segments:
                                process_transcript_segment(segment.text, segment.start, chunk_base_time, audio_start_time)
                                
                            audio_buffer = []
                            is_speaking = False
                            silence_counter = 0
                
                elif transcription_model_name == "mega-asr":
                    audio_buffer.append(audio_data)
                    if len(audio_buffer) >= MAX_CHUNK_LIMIT:
                        full_audio = np.concatenate(audio_buffer)
                        megaASR = _import_mega_asr()
                        if megaASR:
                            print(f"[QONTEX DEBUG] mega-asr: Inferring on {len(full_audio)} samples...")
                            result = megaASR.infer_from_memory(full_audio)
                            print(f"[QONTEX DEBUG] mega-asr: Result type={type(result)}, value={result}")
                            chunk_base_time = (total_chunks - len(audio_buffer)) * CHUNK_DURATION
                            segments = _mega_asr_segments(result)
                            if segments:
                                for text, start in segments:
                                    process_transcript_segment(text, start, chunk_base_time, audio_start_time)
                            else:
                                print(f"[QONTEX DEBUG] mega-asr: Unhandled result format: {type(result)}")
                        audio_buffer = []

            # Flush and transcribe any remaining audio when the stream ends
            if len(audio_buffer) > 0:
                full_audio = np.concatenate(audio_buffer)
                
                chunk_base_time = (total_chunks - len(audio_buffer)) * CHUNK_DURATION
                
                if transcription_model_name == "faster-whisper":
                    segments, _ = model.transcribe(full_audio, beam_size=5, vad_filter=False)
                    for segment in segments:
                        process_transcript_segment(segment.text, segment.start, chunk_base_time, audio_start_time)
                elif transcription_model_name == "mega-asr":
                    megaASR = _import_mega_asr()
                    if megaASR:
                        print(f"[QONTEX DEBUG] mega-asr (flush): Inferring on {len(full_audio)} samples...")
                        result = megaASR.infer_from_memory(full_audio)
                        print(f"[QONTEX DEBUG] mega-asr (flush): Result type={type(result)}, value={result}")
                        segments = _mega_asr_segments(result)
                        if segments:
                            for text, start in segments:
                                process_transcript_segment(text, start, chunk_base_time, audio_start_time)
                        else:
                            print(f"[QONTEX DEBUG] mega-asr (flush): Unhandled result format: {type(result)}")
                
        except Exception as e:
            print(f"[QONTEX] Error in capture loop: {e}")
        finally:
            audio_process.terminate()
            audio_process.wait()

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
