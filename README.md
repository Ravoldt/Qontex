# Qontex

Qontex is a stream context agent for Twitch streams and local VODs. It captures stream audio, transcribes streamer speech, ingests live or replayed chat, writes a chronological timeline, detects likely information requests, and can answer questions from recent stream context through a configurable QA agent.

The current default transcription backend is Mega-ASR. `faster-whisper` is still available as a fallback backend.

## Current Branch Changes

This branch includes the major changes made since the last pushed `main` baseline:

- Added `config.toml` plus ignored `local.toml` overrides for shared and machine-specific settings.
- Added `install_dependencies.py` with automatic PyTorch CUDA wheel selection.
- Extended the installer to install `Mega-ASR/requirements.txt` when the vendored Mega-ASR folder exists.
- Added Mega-ASR support through `TRANSCRIPTION_MODEL = "mega-asr"` and `MEGA_ASR_PATH`.
- Kept `faster-whisper` support through `TRANSCRIPTION_MODEL = "faster-whisper"`.
- Replaced the old `streamlink.py` flow with `streamcapture.py`.
- Added `StreamConfig`, `StreamState`, `Message`, and `SharedDeque` state objects in `stream_state.py`.
- Added live Twitch EventSub chat with stream online/offline handling.
- Added Twitch token validation and refresh helpers.
- Added local VOD mode with synchronized local chat replay.
- Added `PROCESS_FAST` for fast local video transcription.
- Added a QA adapter factory with support for Gemini, disabled QA, and custom `module:ClassName` agents.
- Updated Gemini QA to use recent timeline context, optional image/audio context, Google Search grounding, direct console asks, and optional answer logging.
- Added optional visual frame capture and audio context buffers.
- Added configurable question detector behavior, including a standalone detector hook in `dev/question_detection_pipeline.py`.
- Added runtime config reload and slash commands for updating config values.
- Added delayed sorted JSON timeline flushing to keep chat/transcript order stable.
- Updated `.gitignore` to keep secrets, local settings, logs, temp files, virtualenvs, and Mega-ASR model checkpoints out of Git.

## Project Structure

```text
.
|-- main.py                  # Main runtime entry point and command loop
|-- streamcapture.py         # Audio/video capture, VAD segmentation, ASR backends
|-- twitch_chat.py           # Twitch EventSub chat and local chat replay
|-- stream_state.py          # Runtime config, state, message, and timeline classes
|-- qa_agent.py              # QA adapter interface and factory
|-- gemini_agent.py          # Built-in Gemini QA implementation
|-- utils.py                 # Config, logging, token refresh, question detection
|-- install_dependencies.py  # Dependency installer and PyTorch CUDA selector
|-- config.toml              # Shared runtime defaults
|-- local.toml               # Optional local overrides, ignored by Git
|-- requirements.txt         # Qontex Python requirements
|-- Mega-ASR/                # Vendored modified Mega-ASR source
|-- index.html               # Static dashboard prototype
`-- styles.css               # Dashboard styles
```

## Requirements

- Python 3.11 or newer.
- `ffmpeg` available on `PATH`.
- Streamlink CLI available on `PATH` for Twitch livestreams.
- CUDA-compatible NVIDIA GPU strongly recommended for transcription and classifier models.
- Twitch EventSub credentials for live Twitch chat.
- Gemini API key when `QA_AGENT = "gemini"`.
- Hugging Face access may be needed for some model downloads or private/rate-limited model access.

CPU execution is possible for some paths, but it is expected to be slow. Mega-ASR is intended for GPU use.

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python install_dependencies.py
```

The installer:

- detects CUDA from `nvidia-smi`, `nvcc`, CUDA environment variables, and common CUDA install paths
- installs PyTorch, TorchVision, and TorchAudio from the selected PyTorch wheel index
- installs `requirements.txt`
- installs `Mega-ASR/requirements.txt` when the `Mega-ASR` folder exists
- skips Mega-ASR's pinned `torch`, `torchvision`, and `torchaudio` lines so they do not overwrite the selected PyTorch wheel
- verifies the installed PyTorch/CUDA state

Useful installer options:

```powershell
python install_dependencies.py --cpu
python install_dependencies.py --cuda cu126
python install_dependencies.py --skip-mega-asr
python install_dependencies.py --dry-run
```

Confirm system tools are available:

```powershell
ffmpeg -version
streamlink --version
```

## Mega-ASR Setup

Mega-ASR source is vendored under `Mega-ASR/` so a fresh clone can run without an extra source clone. The model checkpoints are not committed.

Download Mega-ASR weights after installing dependencies:

```powershell
python Mega-ASR\scripts\download.py
```

Qontex expects the default checkpoint layout:

```text
Mega-ASR/
`-- ckpt/
    `-- Mega-ASR/
        |-- Qwen3-ASR-1.7B/
        |-- mega-asr-merged/
        `-- audio_quality_router/
            `-- best_acc_model.safetensors
```

These files are large and intentionally ignored by Git:

```gitignore
Mega-ASR/ckpt/
Mega-ASR/**/.cache/
Mega-ASR/**/*.safetensors
Mega-ASR/**/*.bin
Mega-ASR/**/*.pt
```

If you keep Mega-ASR outside this repo, set `MEGA_ASR_PATH` in `local.toml` to the Mega-ASR repo root or its `src` folder. If the checkpoints are outside the default `Mega-ASR/ckpt/Mega-ASR` path, set `MEGA_ASR_CKPT_DIR` in the environment.

Optional Mega-ASR environment overrides:

```env
MEGA_ASR_CKPT_DIR=C:\path\to\ckpt\Mega-ASR
MEGA_ASR_DEVICE_MAP=auto
MEGA_ASR_ROUTING=true
MEGA_ASR_THRESHOLD=0.5
```

To use the older Whisper path instead:

```toml
TRANSCRIPTION_MODEL = "faster-whisper"
```

## Environment

Create `.env` in the project root:

```env
GENAI_API_KEY=your_gemini_api_key
TWITCH_TOKEN=your_twitch_access_token
TWITCH_REFRESH_TOKEN=your_twitch_refresh_token
TWITCH_CLIENT_ID=your_twitch_client_id
TWITCH_CLIENT_SECRET=your_twitch_client_secret
TWITCH_BOT_ID=your_bot_user_id
HF_TOKEN=optional_huggingface_token
```

Live Twitch chat uses TwitchIO EventSub. `TWITCH_TOKEN` must belong to `TWITCH_BOT_ID`, match `TWITCH_CLIENT_ID`, and include these scopes:

```text
user:read:chat user:bot
```

If your refresh token was authorized before these scopes were added, re-authorize the Twitch app and replace both `TWITCH_TOKEN` and `TWITCH_REFRESH_TOKEN`. Qontex can refresh an expired token, but a refresh token cannot gain scopes it was not originally granted.

`GENAI_API_KEY` is only required when `QA_AGENT = "gemini"` and QA is enabled.

## Configuration

`config.toml` stores shared defaults. Put machine-specific values in `local.toml`; it overrides matching keys from `config.toml` and is ignored by Git.

Minimal live Twitch example:

```toml
CHANNEL = "some_streamer"
SOURCE_TYPE = "livestream"
TWITCH_USERNAME = "your_twitch_username"
TRANSCRIPTION_MODEL = "mega-asr"
ENABLE_QA = true
QA_AGENT = "gemini"
```

Minimal local VOD example:

```toml
CHANNEL = "video/example.mp4"
SOURCE_TYPE = "vod"
CHAT_FILE = "video/example.json"
PROCESS_FAST = true
ENABLE_QA = false
```

Important options:

- `CHANNEL`: Twitch channel name, Twitch URL, or local video path.
- `SOURCE_TYPE`: `auto`, `livestream`, or `vod`.
- `CHAT_FILE`: Optional local chat JSON/JSONL file for VOD mode.
- `TWITCH_USERNAME`: Account label used by the chat listener.
- `TRANSCRIPTION_MODEL`: `mega-asr` or `faster-whisper`.
- `MEGA_ASR_PATH`: Optional path to external Mega-ASR source.
- `PROCESS_FAST`: Runs transcription capture as fast as possible for local videos.
- `ENABLE_QUESTION_DETECTOR`: Enables automatic information-request detection.
- `QUESTION_DETECTOR_TYPE`: `standard` or `standalone`.
- `ENABLE_QA`: Enables sending detected questions to the QA agent.
- `QA_AGENT`: `gemini`, `none`, or `module:ClassName`.
- `QA_MODEL`: Model name passed to the configured QA backend.
- `ENABLE_QA_CHAT`: Allows chat questions to trigger QA.
- `ENABLE_QA_TRANSCRIPT`: Allows transcript questions to trigger QA.
- `ENABLE_ITEMS`: Runs periodic item extraction through the QA agent.
- `QA_CONTEXT_WINDOW`: Recent stream-context window in seconds.
- `ENABLE_VISUAL_CONTEXT`: Sends recent frames to the QA agent.
- `VISUAL_CONTEXT_MAX_FRAMES`: Maximum frame count sent per QA call.
- `VISUAL_CONTEXT_FPS`: Frame capture rate.
- `ENABLE_AUDIO_CONTEXT`: Sends recent audio to the QA agent.
- `AUDIO_CONTEXT_WINDOW`: Audio context window in seconds.
- `LOG_ANSWERS_SEPARATELY`: Also writes QA answers to `answered_questions.json`.
- `LOG_QUESTION_DETECTIONS`: Prints question detection events.
- `FILTER_SHORT_QUESTIONS`: Ignores very short chat questions before classification.
- `SHORT_QUESTION_THRESHOLD`: Word-count cutoff for the short-question filter.
- `LOG_BUFFER_DELAY`: Delay before sorted JSON timeline entries are flushed.

When `PROCESS_FAST = true`, Qontex disables QA, item processing, visual context, and audio context at runtime. This prevents a fast VOD run from making unexpected API calls or accumulating extra media processing.

## Live Twitch Mode

Run the main agent:

```powershell
python main.py
```

For livestreams, Qontex connects to Twitch chat/EventSub first. It waits for the stream to be live before loading AI models and starting capture. When a stream goes offline, Qontex stops capture, flushes logs, unloads AI models from VRAM, and keeps the chat/EventSub listener ready for the next online event.

For testing only Twitch chat:

```powershell
python main.py --test-chat
```

For testing capture/transcription only:

```powershell
python main.py --test-capture
```

Run without QA:

```powershell
python main.py --no-gemini
```

## Local VOD Mode

Set `CHANNEL` to a local video file and use `SOURCE_TYPE = "vod"` or `SOURCE_TYPE = "auto"`. Qontex looks for a chat file beside the video if `CHAT_FILE` is blank:

```text
video/example.mp4
video/example.json
```

Supported chat input shapes:

- TwitchDownloader JSON with a `comments` array.
- JSON lists of chat records.
- JSON-lines records.

Local chat replay is synchronized with the current transcription timestamp. The transcript loop waits for local chat replay when needed so `merged.json` stays chronological.

When TwitchDownloader metadata is present, Qontex uses the VOD streamer name and stream start date for the log folder.

## QA Agents

The built-in Gemini adapter is configured with:

```toml
ENABLE_QA = true
QA_AGENT = "gemini"
QA_MODEL = "gemini-3-flash-preview"
```

Set `QA_AGENT = "none"` to disable QA without using `--no-gemini`.

To use a custom adapter:

```toml
QA_AGENT = "local_agent:LocalAgent"
```

The class may accept `state` or `state=` in its constructor. Required methods:

```python
def answer_question(self, message): ...
def direct_ask(self, question, timestamp=None): ...
```

Optional methods:

```python
def process_items(self): ...
def refresh_from_state(self): ...
def set_game_name(self, game_name): ...
def unload_models(self): ...
```

Gemini answers are written to `merged.json`, and to `answered_questions.json` when `LOG_ANSWERS_SEPARATELY = true`.

## Question Detection

The standard detector uses fast text rules plus a zero-shot classifier for answerable information requests. It handles chat and transcript messages differently, including short-chat filtering and a long-message filter for chat.

To use a standalone detector:

```toml
QUESTION_DETECTOR_TYPE = "standalone"
```

Then provide:

```text
dev/question_detection_pipeline.py
```

with:

```python
def preload_classifier(): ...
def is_likely_question(message, msg_type): ...
```

The `dev/` folder is ignored by Git.

## Visual And Audio Context

Visual context captures frames from the video stream and attaches a bounded set of JPEG frames to Gemini calls:

```toml
ENABLE_VISUAL_CONTEXT = true
VISUAL_CONTEXT_FPS = 1.0
VISUAL_CONTEXT_MAX_FRAMES = 5
```

Audio context stores recent 16 kHz mono PCM chunks and can attach a WAV part to Gemini calls:

```toml
ENABLE_AUDIO_CONTEXT = true
AUDIO_CONTEXT_WINDOW = 60
```

Both options increase memory usage and API payload size. They are disabled automatically in `PROCESS_FAST` mode.

## Runtime Commands

While `main.py` is running, the console accepts:

- `status` or `list`: Show active modules, source state, and queued questions.
- `timeline`: Flush and print the current chronological timeline.
- `clear`: Clear queued detected questions.
- `reload`: Reload `config.toml` and `local.toml`; capture restarts when capture-related settings change.
- `/<config_key> <value>`: Update a config value in `local.toml` if present, otherwise `config.toml`, then reload.
- `/ask <question>`: Ask the QA agent directly with current context.
- `quit`: Stop workers, flush logs, and end the session.

Example:

```text
/ENABLE_QA false
/TRANSCRIPTION_MODEL faster-whisper
/ask what item did the streamer just pick up?
```

## Logs

Qontex writes logs under:

```text
logs/<streamer-or-source>/<YYYY-MM-DD>/
```

Common files:

- `session.log`: Start/stop events and uptime.
- `chat.log`: Live or replayed chat messages.
- `transcript.log`: Transcribed streamer speech.
- `merged.json`: Sorted JSON-lines timeline of chat, transcript, and QA entries.
- `answered_questions.json`: Separate QA answer log when enabled.
- `collected_items.json`: Item extraction output when enabled.

`merged.json` is buffered briefly and flushed in timestamp order to reduce chat/transcript ordering issues.

## Git And Large Files

Commit the modified Mega-ASR source if you want the lowest-setup install path. Do not commit downloaded model weights, caches, secrets, local config, logs, videos, or virtual environments.

The important ignored files and folders are:

```text
.env
local.toml
.tio.tokens.json
.venv/
logs/
video/
lore/
dev/
Mega-ASR/ckpt/
Mega-ASR/**/*.safetensors
Mega-ASR/**/*.bin
Mega-ASR/**/*.pt
```

Mega-ASR is third-party Apache-2.0 code. The license text is included at `Mega-ASR/LICENSE`; keep that license text and attribution with the vendored source, and make it clear that local modifications are part of this project.


`index.html` and `styles.css` are a static dashboard prototype. They are not wired to the Python runtime.

## Troubleshooting

If Mega-ASR import fails, run:

```powershell
python install_dependencies.py
python Mega-ASR\scripts\download.py
```

If Qontex cannot find Mega-ASR, keep the vendored folder at `Mega-ASR/` or set `MEGA_ASR_PATH` in `local.toml`.

If Qontex cannot find weights, run the Mega-ASR download script or set `MEGA_ASR_CKPT_DIR`.

If Twitch chat fails, re-check token ownership, client ID, bot user ID, and scopes. Tokens must include `user:read:chat` and `user:bot`.

If livestream capture fails, confirm `streamlink --version` and `ffmpeg -version` both work in the same shell used to run Qontex.
