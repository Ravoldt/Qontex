# Qontex

Qontex is a stream context agent for Twitch streams and local video files. It captures stream audio, transcribes speech with Whisper, listens to Twitch chat, detects likely questions, and can use Gemini to answer questions from recent stream context. It also writes structured timeline logs that combine transcript, chat, and AI responses. *Currently the question detection a bit hit or miss.

*Transrcibing a stream and syncing the transcription with chat messages works really well.*

## Features

- Captures audio from Twitch streams or local video files with `ffmpeg`.
- Resolves Twitch stream URLs through the Streamlink CLI.
- Transcribes speech with `faster-whisper`.
- Uses Silero VAD to segment speech before transcription.
- Listens to Twitch chat with TwitchIO.
- Detects likely questions from chat and transcript text.
- Sends recent context to Gemini for concise answers.
- Optionally detects collected in-game items from recent transcript/chat context.
- Logs stream sessions under `logs/<channel>/<date>/`.
- Includes a static prototype dashboard in `index.html`.

## Project Structure

```text
.
|-- main.py              # Main runtime entry point
|-- streamlink.py        # Audio/video capture and transcription loop
|-- twitch_chat.py       # Twitch chat listener
|-- gemini_agent.py      # Gemini question-answering and item processing
|-- utils.py             # Config, logging, question detection, shared message buffer
|-- config.json          # Runtime configuration
|-- requirements.txt     # Python dependency seed
|-- index.html           # Static dashboard prototype
`-- styles.css           # Dashboard styles
```

## Requirements

- Python 3.10 or newer
- `ffmpeg` available on your system `PATH`
- Streamlink CLI available on your system `PATH`
- A Twitch OAuth token with chat read access
- A Gemini API key
- CUDA-compatible GPU recommended for Whisper, VAD, and classifier performance

The current code imports these Python packages:

```text
twitchio
python-dotenv
google-generativeai
opencv-python
Pillow
numpy
torch
faster-whisper
transformers
streamlink
```

Install the listed packages into your virtual environment. `requirements.txt` currently only pins TwitchIO, so you may need to install the additional runtime dependencies manually.

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
pip install python-dotenv google-generativeai opencv-python Pillow numpy torch faster-whisper transformers streamlink
```

3. Install system tools.

Make sure these commands work from the terminal:

```powershell
ffmpeg -version
streamlink --version
```

4. Create a `.env` file in the project root.

```env
GENAI_API_KEY=your_gemini_api_key
TWITCH_TOKEN=your_twitch_oauth_token
TWITCH_USERNAME=your_twitch_username
TWITCH_CLIENT_ID=optional_client_id_for_token_refresh
TWITCH_CLIENT_SECRET=optional_client_secret_for_token_refresh
TWITCH_REFRESH_TOKEN=optional_refresh_token
```

`TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET`, and `TWITCH_REFRESH_TOKEN` are optional, but enable automatic token refresh when the current token expires.

## Configuration

Edit `config.json` before running:

```json
{
  "CHANNEL": "streamer_name_or_local_video_path",
  "TWITCH_USERNAME": "your_twitch_username",
  "PROCESS_FAST": false,
  "ENABLE_QA": true,
  "ENABLE_ITEMS": false,
  "QA_CONTEXT_WINDOW": 60,
  "ENABLE_VISUAL_CONTEXT": false,
  "LOG_QUESTION_DETECTIONS": true,
  "FILTER_SHORT_QUESTIONS": false,
  "SHORT_QUESTION_THRESHOLD": 2
}
```

Key options:

- `CHANNEL`: Twitch channel name, `#channel`, Twitch URL source, or local video file path.
- `TWITCH_USERNAME`: Twitch account used by the chat listener.
- `PROCESS_FAST`: For local videos, process as fast as possible instead of real time.
- `ENABLE_QA`: Send detected questions to Gemini.
- `ENABLE_ITEMS`: Periodically ask Gemini to identify collected items from recent context.
- `QA_CONTEXT_WINDOW`: Number of seconds around a question to include as context.
- `ENABLE_VISUAL_CONTEXT`: Intended switch for visual context support.
- `LOG_QUESTION_DETECTIONS`: Print question detection events.
- `FILTER_SHORT_QUESTIONS`: Ignore very short chat questions when enabled.
- `SHORT_QUESTION_THRESHOLD`: Word-count threshold used by the short-question filter.

## Usage

Run the full agent:

```powershell
python main.py
```

Run without Gemini:

```powershell
python main.py --no-gemini
```

Test Twitch chat only:

```powershell
python main.py --test-chat
```

Test audio/video capture only:

```powershell
python main.py --test-capture
```

While the app is running, the command prompt accepts:

- `status`: Show active modules and queued questions.
- `clear`: Clear the question queue.
- `reload`: Reload `config.json`; restarts workers when capture-related settings change.
- `quit`: Stop workers and end the session.

## Logs

Qontex writes session output to:

```text
logs/<streamer-or-source>/<YYYY-MM-DD>/
```

Common files:

- `session.log`: Session start/stop events.
- `chat.log`: Twitch chat messages.
- `transcript.log`: Transcribed streamer audio.
- `merged.json`: JSON-lines timeline of chat, transcript, and Gemini messages.
- `collected_items.json`: JSON-lines item detection output when item processing is enabled.

## Dashboard Prototype

`index.html` contains a static Twitch embed and chat-style UI mockup. It can be opened directly in a browser for layout testing. The WebSocket integration shown in the script block is a placeholder and is not currently wired to the Python runtime.

## Notes

- First run can take time because Whisper, Silero VAD, and the question classifier may download or load large models.
- CPU execution is supported but expected to be slow.
- `.env`, `logs/`, `video/`, `lore/`, and cache folders are ignored by Git.
- Keep Twitch tokens and Gemini API keys out of source control.
