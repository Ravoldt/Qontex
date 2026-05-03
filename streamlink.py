import subprocess
import time
import numpy as np
import os
import cv2
import threading
from collections import deque
import torch
from faster_whisper import WhisperModel
from utils import Message, create_stream_folder, get_config_value, get_streamer_name, is_likely_question, log_json, log_message, log_start_stop, shared_deque
import datetime
import json

# Exported so it can be set by main.py
STREAM_START_TIME = None

# In-memory buffer for video frames. Capture stores one frame per second, matching SharedDeque's 3-minute window.
video_frames = deque(maxlen=shared_deque.max_age)

_whisper_model = None
_vad_model = None

def preload_models():
    global _whisper_model, _vad_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA is not available. Loading models on CPU (this will be slow).")
        compute_type = "int8"
    else:
        compute_type = "float16"

    if _whisper_model is None:
        print(f"Loading faster-whisper model into {'VRAM' if device == 'cuda' else 'RAM'}... (this may take a moment)")
        _whisper_model = WhisperModel("large-v3", device=device, compute_type=compute_type) 
    
    if _vad_model is None:
        print(f"Loading Silero VAD model into {'VRAM' if device == 'cuda' else 'RAM'}...")
        _vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
        _vad_model = _vad_model.to(device)
        _vad_model.eval()

def start_audio_capture(source, process_fast=False):
    """Starts a background process to extract audio from a video or livestream."""
    if not source.startswith("http") and "." not in source:
        source = f"https://www.twitch.tv/{source}"

    is_livestream = "twitch.tv" in source or source.startswith("http")
    m3u8_url = None
    
    command = ["ffmpeg"]
    
    if not is_livestream and not process_fast:
        command.append("-re")
        
    if is_livestream:
        print(f"Resolving stream URL for {source}...")
        try:
            m3u8_url = subprocess.check_output(
                ["streamlink", "--stream-url", source, "audio_only"],
                stderr=subprocess.STDOUT
            ).decode("utf-8").strip()
            command.extend(["-i", m3u8_url])
        except subprocess.CalledProcessError as e:
            print(f"Error resolving stream. Is the streamer offline?\nDetails: {e.output.decode('utf-8').strip()}")
            return None, None
        except FileNotFoundError:
            print("Error: Streamlink is not installed or not in your system PATH.")
            return None, None
        except Exception as e:
            print(f"Error resolving stream: {e}")
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
        print("Error: Streamlink is not installed or not in your system PATH.")
        return None, None

def capture_video_frames(source_url, stop_event=None):
    """Background thread to capture exactly one frame per second from the video stream."""
    cap = cv2.VideoCapture(source_url)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
        
    frame_interval = max(1, int(fps))
    frame_count = 0
    
    while not (stop_event and stop_event.is_set()):
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            video_frames.append(frame)
            
        frame_count += 1
        
    cap.release()

def run_capture_loop(source, log_folder, process_fast=False, question_handler=None, stop_event=None):
    """Main capture loop running as a background thread."""
    global STREAM_START_TIME, _whisper_model, _vad_model
    transcript_user = get_streamer_name(source)
    
    if _whisper_model is None or _vad_model is None:
        preload_models()
    
    model = _whisper_model
    vad_model = _vad_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    audio_process, m3u8_url = start_audio_capture(source, process_fast=process_fast)
    
    if audio_process:
        target_url = m3u8_url if m3u8_url else source
        video_thread = threading.Thread(target=capture_video_frames, args=(target_url, stop_event), daemon=True)
        video_thread.start()
        
        print("Audio stream captured! Running AI-VAD loop...")
        
        try:                      
            # Silero natively prefers exactly 32ms chunks (512 samples at 16kHz)
            CHUNK_DURATION = 0.032  
            CHUNK_SIZE = int(16000 * 2 * CHUNK_DURATION) 
            
            SPEECH_THRESHOLD = 0.5     # AI Confidence: 50% probability it is human speech
            MAX_SILENCE_CHUNKS = 30    # ~1.0 second of silence triggers the end of a sentence
            MAX_CHUNK_LIMIT = 156      # ~5.0 seconds maximum chunk length (The Safety Valve)
            
            audio_buffer = []
            silence_counter = 0
            is_speaking = False
            
            total_chunks = 0
            audio_start_time = None
            
            while not (stop_event and stop_event.is_set()):
                in_bytes = audio_process.stdout.read(CHUNK_SIZE)
                if not in_bytes:
                    break 

                if audio_start_time is None:
                    audio_start_time = time.time()
                    
                total_chunks += 1

                audio_data = np.frombuffer(in_bytes, np.int16).astype(np.float32) / 32768.0
                
                # Ask Silero if someone is speaking in this 32ms window
                audio_tensor = torch.from_numpy(audio_data).to(device)
                speech_prob = vad_model(audio_tensor, 16000).item()
                
                if speech_prob > SPEECH_THRESHOLD:
                    is_speaking = True
                    silence_counter = 0
                    audio_buffer.append(audio_data)
                elif is_speaking:
                    silence_counter += 1
                    audio_buffer.append(audio_data)
                    
                    # Cut the chunk if we hit a 1-second pause OR if they have talked non-stop for 5 seconds
                    if silence_counter >= MAX_SILENCE_CHUNKS or len(audio_buffer) >= MAX_CHUNK_LIMIT:
                        full_audio = np.concatenate(audio_buffer)
                        
                        # vad_filter=False because our Silero loop already perfectly trimmed the dead air!
                        segments, _ = model.transcribe(full_audio, beam_size=15, vad_filter=False)
                        
                        chunk_base_time = (total_chunks - len(audio_buffer)) * CHUNK_DURATION
                        
                        for segment in segments:
                            if STREAM_START_TIME is None or process_fast:
                                msg_time = chunk_base_time + segment.start
                            else:
                                segment_wall_time = audio_start_time + chunk_base_time + segment.start
                                msg_time = segment_wall_time - STREAM_START_TIME

                            msg_time = max(0.0, msg_time)  
                                
                            msg = Message(msg_time, "transcript", segment.text.strip(), user=transcript_user)
                            print(msg)
                            log_message(log_folder, "transcript.log", msg)
                            log_json(log_folder, "merged.json", msg.to_dict())
                            shared_deque.add_message(msg)
                            should_detect_question = question_handler or get_config_value("LOG_QUESTION_DETECTIONS", True)
                            if should_detect_question and is_likely_question(msg.text, msg.type) and question_handler:
                                threading.Thread(target=question_handler, args=(msg,), daemon=True).start()
                        
                        audio_buffer = []
                        is_speaking = False
                        silence_counter = 0
                
        except Exception as e:
            print(f"Error in capture loop: {e}")
        finally:
            audio_process.terminate()
            audio_process.wait()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    STREAM_START_TIME = time.time()
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("CRITICAL ERROR: config.json is missing!")
        exit(1)
        
    source = config.get("CHANNEL", "")
    is_local_video = os.path.isfile(source)
    
    if not is_local_video and not source.startswith("http"):
        source = source.lstrip('#')
    
    if not source:
        print("Please set the CHANNEL in config.json!")
        exit(1)
        
    process_fast = config.get("PROCESS_FAST", False)

    streamer_name = get_streamer_name(source)
    stream_start = datetime.datetime.now()
    log_folder = create_stream_folder(streamer_name, stream_start)
    log_start_stop(log_folder, "start")
    
    print(f"Testing capture loop for {source}...")
    run_capture_loop(source, log_folder, process_fast=process_fast)
