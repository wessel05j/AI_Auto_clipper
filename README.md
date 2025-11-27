# AI Clipper

Automated pipeline that (1) scans a folder of videos, (2) transcribes each using Whisper, (3) chunks long transcripts, (4) uses an LLM to identify interesting segments based on a user query, (5) merges nearby timestamps, and (6) exports the matching clips as individual MP4 files.

## Features
- Batch video discovery from `videos/`
- Fast transcription via Whisper local models (e.g. `tiny`, `base`, `small`, etc.)
- Transcript chunking to keep prompts within model token limits
- LLM-guided semantic span extraction (local server via LM Studio or OpenAI-compatible endpoint)
- Tolerance-based segment merging to avoid fragmented clips
- Automatic clip rendering to `output/` using MoviePy
- Make sure you have a `system/`, `output/` and a `videos/` folder.

## Directory Structure
```
AI_clipper/
  main.py                # Orchestrates the full pipeline
  yt_downloader.py       # Download youtube videos before starting
  scan_videos.py         # Collects video file paths
  transcribing.py        # Whisper transcription logic
  chunking.py            # Splits transcript into token-sized chunks
  ai_scanning.py         # LLM call to extract relevant spans
  merge_segments.py      # Merges close timestamp segments
  extract_clips.py       # Cuts clips using MoviePy
  variable_checker.py    # Checks if the variables has inputs
  videos/                # Input source videos (place your .mp4 files here)
  output/                # Generated clips
  system/                # Temporary JSON artifacts (transcribed + AI output)
```

## Requirements
Install Python 3.9+ and the following packages:
- `whisper` (openai-whisper)
- `openai` (for OpenAI-compatible client, works with LM Studio local server)
- `moviepy` (for clipping)

Suggested installation:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install openai-whisper moviepy openai
```
If using CUDA for faster Whisper, also install `torch` with the appropriate wheel.

## Configuration (edit `main.py`)
Set these variables near the top of `main.py`:
```python
user_query = "motivational gym advice"        # What you want the AI to find
base_url = "http://localhost:1234/v1"          # LM Studio or other OpenAI-compatible endpoint
model = "TheModelName"                         # e.g. "phi-2", "gpt-4o-mini", local model alias
clips_output = "output/"                       # Folder for exported clips
transcribing_model = "base"                    # Whisper model size (tiny/base/small/...)
```
Other tunables:
- `max_token` inside `main.py` controls chunk sizing (effective input ≈ 60% of this)
- `merge_segments(AI_output, 30)` uses a 30 second tolerance; lower for tighter segmentation

## Usage
1. Place source `.mp4` files into the `videos/` directory.
2. Edit variables in `main.py` (at minimum: `user_query`, `base_url`, `model`).
3. Activate your virtual environment and start any local LLM server if needed.
4. Run:
```powershell
python main.py
```
5. Resulting clips appear incrementally in `output/` (`0.mp4`, `1.mp4`, ...).

Intermediate artifacts:
- `system/transcribed.json` stores the transcript segments for a single video.
- `system/AI.json` stores raw model span outputs before merging.
These are deleted after each video iteration for a clean state.

## How It Works (Pipeline)
1. `scan_videos.py` enumerates files in `videos/`.
2. `transcribing.py` loads Whisper and filters very short text fragments.
3. `chunking.py` groups transcript segments until token threshold reached.
4. `ai_scanning.py` sends each chunk plus your `user_query` to the LLM, requesting only JSON arrays of `[start, stop]` pairs.
5. `merge_segments.py` combines close segments (within tolerance seconds) into longer clips.
6. `extract_clips.py` slices the original video using MoviePy and writes output files.

## Customization Ideas
- Adjust whisper model for speed vs accuracy.
- Add concurrency for multiple videos.
- Extend `ai_scanning.py` with retry logic & rate limiting.
- Add logging instead of print statements.
- Generate human-readable summaries of selected clips.

## Troubleshooting
- Empty output folder: Ensure `user_query` matches actual transcript content.
- Whisper errors: Confirm `ffmpeg` installed (MoviePy & Whisper rely on it).
- LLM returns invalid JSON: Add validation / retry around `json.loads` in `ai_scanning.py`.
- Long runtime: Use smaller Whisper model or GPU acceleration.

## Roadmap
- Config file (YAML/JSON) instead of hard-coded variables.
- Optional sentiment / topic classification for clip ranking.
- CLI arguments (`--query`, `--model`, `--tolerance`, etc.).
- Unit tests for chunking & merging logic.

## License
Licensed under the Apache License, Version 2.0. See the `LICENSE` file for the full text. You may not use this project except in compliance with the License.

Copyright (c) 2025 (Add your name or organization here)

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Disclaimer
This tool depends on transcript + model quality. Always review exported clips before publishing.

Enjoy faster semantic clip extraction!
