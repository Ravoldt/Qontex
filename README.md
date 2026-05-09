# Qontex

Qontex is a stream context agent for Twitch streams and local video files. It captures audio, transcribes streamer speech with `faster-whisper`, ingests Twitch chat, builds a merged chronological timeline, detects likely questions, and can use Gemini to answer questions from recent stream context.

## Features

- Captures Twitch livestream or local video audio through `ffmpeg`.
- Resolves Twitch livestream audio URLs through the Streamlink CLI.
- Transcribes speech with `faster-whisper` using a Silero VAD segmentation loop.
- Listens to live Twitch chat through TwitchIO EventSub.
- Replays matching local chat JSON files alongside local video files.
- Keeps chat and transcript synchronized in `merged.json`.
- Detects likely questions from chat and transcript text when enabled.
- Sends question context to Gemini when QA is enabled.
- Optionally includes recent video frames in Gemini context outside fast mode.
- Logs sessions under `logs/<streamer>/<date>/`.
- Includes a static dashboard prototype in `index.html`.

## Project Structure

```text
.
|-- main.py                 # Main runtime entry point and command loop
|-- streamcapture.py        # Audio extraction, VAD, frame capture, transcription
|-- twitch_chat.py          # Live Twitch chat and local chat replay
|-- gemini_agent.py         # Gemini QA and item-processing helpers
|-- utils.py                # Config, logging, question detection, shared timeline buffer
|-- install_dependencies.py # Dependency installer with PyTorch CUDA wheel selection
|-- config.toml             # Shared runtime configuration
|-- local.toml              # Local override configuration, ignored by Git
|-- requirements.txt        # Python dependency seed
|-- index.html              # Static dashboard prototype
`-- styles.css              # Dashboard styles
```

## Requirements

- Python 3.11 or newer
- `ffmpeg` available on `PATH`
- Streamlink CLI available on `PATH` for Twitch livestreams
- Twitch EventSub chat credentials for live chat
- Gemini API key when Gemini is enabled
- CUDA-compatible GPU recommended for Whisper, VAD, and classifier performance

Install Python dependencies with:

```powershell
python install_dependencies.py
```

The installer chooses a PyTorch wheel based on detected CUDA 12 support. To force CPU-only PyTorch:

```powershell
python install_dependencies.py --cpu
```

You can install the plain requirements directly, but that does not choose a CUDA wheel for you:

```powershell
pip install -r requirements.txt
```

Confirm system tools are available:

```powershell
ffmpeg -version
streamlink --version
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
```

Live Twitch chat uses TwitchIO EventSub. `TWITCH_TOKEN` must belong to `TWITCH_BOT_ID`, match `TWITCH_CLIENT_ID`, and include the `user:read:chat` scope. `TWITCH_REFRESH_TOKEN` is required by the live chat listener and is also used for automatic token refresh.

`GENAI_API_KEY` is only required when Gemini is enabled. It is not required when running with `--no-gemini` or with `PROCESS_FAST = true`.

## Configuration

`config.toml` contains shared defaults. `local.toml` overrides matching values and is the right place for machine-specific paths, test videos, usernames, and fast-mode settings.

```toml
CHANNEL = "streamer_name_or_local_video_path"
TWITCH_USERNAME = "your_twitch_username"
PROCESS_FAST = false
ENABLE_QUESTION_CHECKER = true
ENABLE_QA = true
ENABLE_QA_CHAT = false
ENABLE_QA_TRANSCRIPT = true
ENABLE_ITEMS = false
QA_CONTEXT_WINDOW = 60
ENABLE_VISUAL_CONTEXT = false
LOG_ANSWERS_SEPARATELY = true
LOG_QUESTION_DETECTIONS = true
FILTER_SHORT_QUESTIONS = false
SHORT_QUESTION_THRESHOLD = 2
LOG_BUFFER_DELAY = 20
```

Key options:

- `CHANNEL`: Twitch channel name, `#channel`, Twitch URL, or local video path.
- `TWITCH_USERNAME`: Twitch account label used by the chat listener.
- `PROCESS_FAST`: For local videos, processes as fast as possible instead of realtime.
- `ENABLE_QUESTION_CHECKER`: Enables heuristic and classifier-based question detection.
- `ENABLE_QA`: Allows detected questions to be sent to Gemini.
- `ENABLE_QA_CHAT`: Enables Gemini answers for chat questions when QA is enabled.
- `ENABLE_QA_TRANSCRIPT`: Enables Gemini answers for transcript questions when QA is enabled.
- `ENABLE_ITEMS`: Periodically asks Gemini to identify collected items from recent context.
- `QA_CONTEXT_WINDOW`: Number of seconds of context used for Gemini answers.
- `ENABLE_VISUAL_CONTEXT`: Sends recent captured frames to Gemini when enabled.
- `LOG_ANSWERS_SEPARATELY`: Writes answered questions to `answered_questions.json`.
- `LOG_QUESTION_DETECTIONS`: Prints question detection events.
- `FILTER_SHORT_QUESTIONS`: Ignores very short chat questions when enabled.
- `SHORT_QUESTION_THRESHOLD`: Word-count cutoff for short-question filtering.
- `LOG_BUFFER_DELAY`: Seconds to buffer JSON timeline records before writing sorted batches.

## Fast Local Processing

`PROCESS_FAST = true` is intended for local video files. In fast mode:

- `ffmpeg` does not use realtime input throttling.
- Gemini QA, chat QA, transcript QA, item processing, and visual context are forced off at runtime.
- Manual `/ask` is blocked so a fast run cannot accidentally hit the Gemini API.
- Video frame capture is disabled.
- Matching local chat JSON is still replayed and synchronized with transcription.

For a local video such as:

```toml
CHANNEL = "video/example.mp4"
```

Qontex looks for:

```text
video/example.json
```

Supported local chat inputs include TwitchDownloader-style JSON with a `comments` array, JSON lists, and JSON-lines records. Messages are sorted by timestamp with a stable sort, so messages that share the same timestamp keep their original file order. The transcript loop waits for local chat replay to catch up, keeping `merged.json` in chronological order.

When a local TwitchDownloader JSON file includes streamer and video metadata, Qontex uses it to choose the log folder streamer name and stream start date.

## Usage

Run the full agent:

```powershell
python main.py
```

Run without Gemini:

```powershell
python main.py --no-gemini
```

Test live Twitch chat only:

```powershell
python main.py --test-chat
```

Test capture/transcription only:

```powershell
python main.py --test-capture
```

While running, the command prompt accepts:

- `status` or `list`: Show active modules and queued questions.
- `timeline`: Flush and print the current merged timeline.
- `clear`: Clear the question queue.
- `reload`: Reload `config.toml` and `local.toml`; restarts workers when capture-related settings change.
- `/<config_key> <value>`: Update a setting in `config.toml` and trigger reload, for example `/ENABLE_QA false`.
- `/ask <question>`: Send a direct question to Gemini, unless fast mode or `--no-gemini` is active.
- `quit`: Stop workers, flush logs, and end the session.

## Logs

Qontex writes output to:

```text
logs/<streamer-or-source>/<YYYY-MM-DD>/
```

Common files:

- `session.log`: Session start and stop events.
- `chat.log`: Live or replayed chat messages.
- `transcript.log`: Transcribed streamer audio.
- `merged.json`: JSON-lines timeline of chat, transcript, and Gemini messages.
- `answered_questions.json`: Gemini answers when separate answer logging is enabled.
- `collected_items.json`: Item detection output when item processing is enabled.

`merged.json` writes through a small buffer so chat and transcript entries can be sorted by timestamp before being appended.

## Dashboard Prototype

`index.html` contains a static Twitch embed and chat-style UI mockup. It can be opened directly in a browser for layout testing. The WebSocket code in the page is placeholder code and is not wired to the Python runtime.

## Notes

- First run can take time because Whisper, Silero VAD, and the question classifier may download or load large models.
- CPU execution is supported but expected to be slow.
- `PROCESS_FAST` is safest for local offline transcription because it disables Gemini and frame capture.
- Keep Twitch tokens and Gemini API keys out of source control.

