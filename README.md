# AI Auto Clipper

AI Auto Clipper is a local-first pipeline that turns long videos into short, query-matched clips.
It transcribes with Whisper, scans transcript chunks with an Ollama model, merges nearby matches, and exports MP4 clips.

## What You Get
- Local transcription with Whisper (`tiny` to `large`)
- Local semantic clip selection through Ollama
- Optional YouTube ingest (single links and channel monitoring)
- Resume support through temp progress files
- Automatic `torch` setup with CUDA detection (and CPU fallback)

## Quick Start (Windows)
1. Install Python 3.10+ and make sure `python` works in `cmd`.
2. Install FFmpeg and add it to `PATH`.
3. Install Ollama and pull a model (example: `ollama pull llama3.2`).
4. Run `settings.bat` and complete the setup wizard.
5. Put video files in `input/`.
6. Run `main.bat`.
7. Find clips in `output/`.

## Quick Start (macOS/Linux)
1. Install Python 3.10+, FFmpeg, and Ollama.
2. In the project root, run:

```bash
python -m venv venv
source venv/bin/activate
python setup_env.py --torch auto
python settings.py
python main.py
```

## Installation Notes

### Required External Tools
- Python 3.10+
- FFmpeg (needed by MoviePy/Whisper)
- Ollama server + at least one pulled model

### Python Dependencies
`setup_env.py` installs:
- `openai-whisper`
- `moviepy==1.0.3`
- `yt_dlp`
- `requests`
- `tiktoken`
- `ollama`
- `tqdm`
- `numpy<2`
- `torch` (auto CPU/CUDA mode)

### CUDA Automation
You do not need to manually install a CUDA torch wheel anymore for normal usage.
`setup_env.py --torch auto` does:
1. Detects whether NVIDIA GPU tooling is available.
2. Tries CUDA torch wheels.
3. Falls back to CPU torch if CUDA is unavailable.

Optional overrides:

```bash
python setup_env.py --torch cuda
python setup_env.py --torch cpu
python setup_env.py --torch skip
python setup_env.py --torch auto --force
```

Validation command:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## Run Commands
- Configure: `settings.bat` (or `python settings.py`)
- Clip videos: `main.bat` (or `python main.py`)
- Fetch recent channel videos into queue: `fetch_yt_links.bat` (or `python fetch_yt_links.py`)

All three `.bat` files route through `run_with_venv.bat`, which:
1. Creates/activates `venv` if needed
2. Runs `setup_env.py`
3. Executes the requested script

## Project Structure
```text
AI_Auto_clipper/
  main.py
  settings.py
  fetch_yt_links.py
  setup_env.py
  run_with_venv.bat
  input/
  output/
  system/
  temp/
  programs/
    components/
    core_functionality/
```

## Workflow
1. Scan files in `input/`.
2. Transcribe video with Whisper.
3. Chunk transcript to fit token budget.
4. Ask Ollama for matching clip timestamps.
5. Merge nearby timestamps.
6. Extract clips to `output/`.
7. Move processed source video to `temp/`.

## YouTube Intake

### Add Direct Video Links
Use `settings.py` option `6` and add links to `youtube_list`.

### Monitor Channels
1. In `settings.py`, set channel URLs in option `11`.
2. Set hour window in option `12`.
3. Run `fetch_yt_links.py` to add newly found links into `youtube_list`.

### Cookies
If YouTube blocks downloads, place `cookies.txt` in one of:
- `resources/cookies.txt`
- `system/cookies.txt`
- project root `cookies.txt`

## Key Config Fields (`system/settings.json`)
- `ai_model`: Ollama model tag
- `transcribing_model`: Whisper model size
- `user_query`: what clips to search for
- `system_query`: strict output contract for model responses
- `total_tokens`: model context budget
- `max_chunking_tokens`: computed chunk budget
- `max_ai_tokens`: reserved response budget
- `merge_distance`: merge nearby clips by seconds
- `ai_loops`: repeated scans per chunk
- `temperature`: creativity/randomness
- `rerun_temp_files`: resume behavior toggle

## Troubleshooting
- `Settings file not found`: run `settings.py` once first.
- `ffmpeg` errors: install FFmpeg and verify `ffmpeg -version`.
- No clips found: tighten `user_query`, increase `ai_loops`, or lower `temperature`.
- Ollama connection issues: verify `ollama serve` and `ollama_url` in settings.
- Slow runtime: use smaller Whisper model (`tiny`/`base`) or enable CUDA.
- Download issues: refresh `cookies.txt` and retry.

## License
Apache License 2.0 (`LICENSE`).
