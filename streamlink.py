import subprocess
import time
import numpy as np
from faster_whisper import WhisperModel


def start_audio_capture(source, process_fast=False):
    """Starts a background process to extract audio from a video or livestream."""
    # If source is just a username (no URL, no file extension), convert it to a Twitch URL
    if not source.startswith("http") and "." not in source:
        source = f"https://www.twitch.tv/{source}"

    is_livestream = "twitch.tv" in source or source.startswith("http")
    
    command = ["ffmpeg"]
    
    # If it's a local file and we want real-time speed, use -re
    if not is_livestream and not process_fast:
        command.append("-re")
        
    if is_livestream:
        print(f"Resolving stream URL for {source}...")
        try:
            # Ask streamlink for the direct HLS stream URL (audio_only)
            m3u8_url = subprocess.check_output(
                ["streamlink", "--stream-url", source, "audio_only"],
                stderr=subprocess.STDOUT
            ).decode("utf-8").strip()
            command.extend(["-i", m3u8_url])
        except subprocess.CalledProcessError as e:
            print(f"Error resolving stream. Is the streamer offline?\nDetails: {e.output.decode('utf-8').strip()}")
            return None
        except FileNotFoundError:
            print("Error: Streamlink is not installed or not in your system PATH.")
            return None
        except Exception as e:
            print(f"Error resolving stream: {e}")
            return None
    else:
        command.extend(["-i", source])
        
    command.extend([
        "-vn", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"
    ])
    
    try:
        # Launch the background process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,     # Capture the audio stream
            stderr=subprocess.DEVNULL   # Ignore streamlink's internal connection logs
        )
        return process
        
    except FileNotFoundError:
        print("Error: Streamlink is not installed or not in your system PATH.")
        return None

# --- Main Execution ---

if __name__ == "__main__":
    # Set this to a Twitch URL OR a local video file path
    source = "https://www.twitch.tv/babylon340"
    process_fast = False  # Set to False to simulate real-time on local videos

    print("Loading faster-whisper model... (this may take a moment)")
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cuda", compute_type="float16") 
    
    audio_process = start_audio_capture(source, process_fast=process_fast)
    
    if audio_process:
        print("Audio stream captured! Running VAD loop...")
        
        try:                      
            CHUNK_DURATION = 0.5  # Read 0.5 seconds of audio at a time
            CHUNK_SIZE = int(16000 * 2 * CHUNK_DURATION) # 16kHz * 16-bit (2 bytes)
            
            SILENCE_THRESHOLD = 0.005  # Volume threshold (adjust if needed)
            MAX_SILENCE_CHUNKS = 2     # How many silent chunks (1 sec) equal a pause?
            
            audio_buffer = []
            silence_counter = 0
            is_speaking = False

            stream_time = 0.0       # Track total time processed
            buffer_start_time = 0.0 # Track when the current spoken phrase started        
            
            while True:
                in_bytes = audio_process.stdout.read(CHUNK_SIZE)
                if not in_bytes:
                    break # Stream has ended

                stream_time += CHUNK_DURATION
            
                    
                # Convert the raw bytes to a float32 numpy array normalized to [-1.0, 1.0]
                audio_data = np.frombuffer(in_bytes, np.int16).astype(np.float32) / 32768.0
                
                # Calculate the volume (mean absolute amplitude) of this chunk
                volume = np.abs(audio_data).mean()
                
                if volume > SILENCE_THRESHOLD:
                    if not is_speaking:
                        buffer_start_time = stream_time - CHUNK_DURATION                
                    is_speaking = True
                    silence_counter = 0
                    audio_buffer.append(audio_data)
                elif is_speaking:
                    silence_counter += 1
                    audio_buffer.append(audio_data)
                    
                    if silence_counter >= MAX_SILENCE_CHUNKS:
                        # We hit a pause. Combine chunks and transcribe!
                        full_audio = np.concatenate(audio_buffer)
                        
                        # vad_filter=True utilizes faster-whisper's built-in Silero VAD to clean internal dead air
                        segments, _ = model.transcribe(full_audio, beam_size=5, vad_filter=True)
                        
                        for segment in segments:
                            abs_start = buffer_start_time + segment.start
                            abs_end = buffer_start_time + segment.end
                            print(f"[{abs_start:.2f}s -> {abs_end:.2f}s] {segment.text.strip()}")
                        
                        # Reset the buffer for the next sentence
                        audio_buffer = []
                        is_speaking = False
                        silence_counter = 0
                
        except KeyboardInterrupt:
            print("\nStopping stream capture...")
            # Always clean up background processes when your script exits!
            audio_process.terminate()
            audio_process.wait()
            print("Cleanup complete.")