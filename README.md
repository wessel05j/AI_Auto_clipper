# AI Auto Clipper

Automated pipeline that
1. discovers local or YouTube videos,
2. transcribes them with Whisper,
3. chunks long transcripts using token estimation,
4. sends each chunk to a local LLM via Ollama to find interesting segments based on a user query,
5. merges nearby timestamps, and
6. exports the matching clips as individual MP4 files.

The whole thing is started from `main.py` and configured interactively – you do **not** have to edit code to use it.

---

## Features
- Interactive setup wizard that stores settings in `system/settings.json`.
- Batch video discovery from an input folder you choose.
- Optional YouTube download step (via `yt_dlp`) into the input folder.
- Fast transcription via Whisper local models (e.g. `tiny`, `base`, `small`, etc.).
- Token-aware transcript chunking using tiktoken estimation and your model's max tokens.
- LLM-guided semantic span extraction via **Ollama** (local runtime, default at `http://localhost:11434`).
- Tolerance-based segment merging to avoid fragmented clips.
- Automatic clip rendering to your output folder using MoviePy.

---

## Project Layout
```
AI_Auto_clipper/
  main.py                     # Orchestrates setup + full pipeline
  settings.py                 # Interactive settings menu
  programs/
   components/
    file_exists.py          # Tiny helper to check JSON/clip files
    interact_w_json.py      # Read/write JSON helper
    load.py                 # JSON load wrapper
    write.py                # JSON write wrapper
    return_tokens.py        # Token estimation using tiktoken
    scan_videos.py          # Collects video file paths from input folder
   core_functionality/
    yt_downloader.py        # Downloads YouTube videos to input folder
    transcribing.py         # Whisper transcription + basic segment merging
    chunking.py             # Splits transcript into token-sized chunks
    ai_scanning.py          # LLM call to extract relevant [start, end] spans
    merge_segments.py       # Merges close timestamp segments
    extract_clip.py         # Cuts clips using MoviePy
    ollama_on.py            # Starts Ollama server
    ollama_off.py           # Stops Ollama server
    ollama_chat.py          # Chat with Ollama
    ollama_scanning.py      # Scanning with Ollama
  system/                     # Settings + temporary JSON artifacts (settings.json, transcribing.json, AI.json, clips.json)
  input/                      # Input videos
  output/                     # Generated clips
  temp/                       # Temp files
```

---

## Requirements

### System Dependencies
- **FFmpeg** must be installed and available on `PATH` (required by MoviePy for video processing).
  - Download from https://ffmpeg.org/download.html
  - Add to PATH.
- **Ollama** for local LLMs: https://ollama.com/download
  - Install and pull a model, e.g. `ollama pull llama3.2`

### Python
- Python 3.10–3.11 recommended.

### Python Packages
Install with `pip install -r requirements.txt`:
- `openai-whisper` – Whisper transcription.
- `ollama` – Ollama client.
- `moviepy==1.0.3` – Video clipping (pinned due to compatibility issues with newer versions).
- `yt_dlp` – YouTube downloading.
- `torch==2.0.1` – Required by Whisper (CUDA version for GPU support; requires CUDA 11.8).
- `requests` – HTTP requests.
- `tiktoken` – Token estimation.
- `tqdm` – Progress bars.
- `flask` – Web dashboard.
- `numpy<2` – NumPy (pinned for compatibility).

---

## Running the Program

1. **Install Dependencies**
   - Install FFmpeg and Ollama as above.
   - Run Ollama and pull a model: `ollama pull llama3.2`

4. **First Run**
   - Run `python settings.py`
   - Follow interactive setup: choose model, query, etc.
   - Settings saved to `system/settings.json`

5. **Subsequent Runs**
   - Run `python settings.py` to edit settings.
   - Run `python main.py` to process videos.
   - Optionally, run `python web.py` and open http://localhost:5000 for progress dashboard.

Output clips in `output/`.

---

## Tips
- Use smaller Whisper models for speed.
- Adjust merge distance for clip length.

---

## License
Apache License 2.0
