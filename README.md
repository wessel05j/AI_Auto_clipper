# AI Auto Clipper

Automated pipeline that
1. discovers local or YouTube videos,
2. transcribes them with Whisper,
3. chunks long transcripts using approximate token estimation,
4. sends each chunk to a local LLM via Ollama to find interesting segments based on a user query,
5. merges nearby timestamps, and
6. exports the matching clips as individual MP4 files.

The whole thing is started from `main.py` and configured interactively – you do **not** have to edit code to use it.

---

## Features
- Interactive setup wizard that stores settings in `system/setup_settings.json`.
- Batch video discovery from an input folder you choose.
- Optional YouTube download step (via `yt_dlp`) into the input folder.
- Fast transcription via Whisper local models (e.g. `tiny`, `base`, `small`, etc.).
- Token-aware transcript chunking using approximate token estimation and your model’s max tokens.
- LLM-guided semantic span extraction via **Ollama** (local runtime, default at `http://localhost:11434`).
- Tolerance-based segment merging to avoid fragmented clips.
- Automatic clip rendering to your output folder using MoviePy.

---

## Project Layout
Top-level:
```text
AI_Auto_clipper/
  main.py                     # Orchestrates setup + full pipeline
  programs/
   components/
    file_exists.py          # Tiny helper to check JSON/clip files
    interact_w_json.py      # Read/write JSON helper
    load.py                 # JSON load wrapper
    wright.py               # JSON write wrapper
   core_functionality/
    scan_videos.py          # Collects video file paths from input folder
    yt_downloader.py        # Downloads YouTube videos to input folder
    transcribing.py         # Whisper transcription + basic segment merging
    chunking.py             # Splits transcript into token‑sized chunks
    ai_scanning.py          # LLM call to extract relevant [start, end] spans
    merge_segments.py       # Merges close timestamp segments
    extract_clip.py         # Cuts clips using MoviePy
   setup_stage/
    setup_stage.py          # Interactive first‑run and boot menu
    interact_w_ai.py        # Test LLM connection
    max_tokens_ai_check.py  # Ask the LLM for its max tokens
  system/                     # Settings + temporary JSON artifacts (settings.json, log.txt, run_state.json)
  videos/                     # (Optional) Example input folder for local videos
  output/                     # Generated clips
```

On first run, `system/settings.json` is created automatically by `setup_stage.py`.

---

## Requirements

### Python
- Python **3.10–3.11** recommended. Newer versions (e.g., 3.12/3.13) may have limited binary wheel support for some packages like `torch`/`moviepy` on Windows.

### Python packages
Core runtime dependencies are listed in `requirements.txt`:
- `openai-whisper` – Whisper transcription.
- `ollama` – Local LLM client for chat completion.
- `moviepy==1.0.3` – Video clipping.
- `yt_dlp` – YouTube downloading (optional).
- `torch` – Required by Whisper; install a CPU or GPU build appropriate for your system.

### System tools
- **FFmpeg** must be installed and available on `PATH` (required by MoviePy and some Whisper backends).
- **Ollama** (Windows installer) for local LLMs: https://ollama.com/download
  - After installing, pull your model, e.g.:
    - `ollama pull gpt-oss:20b`
  - Verify it works:
    - `ollama run gpt-oss:20b --prompt "hello"`

#### Recommended installation (Windows / PowerShell)
```powershell
cd d:\Work\Projects\AI_Auto_clipper

python -m venv .venv
\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

# Optional: CPU‑only torch (adjust for your system)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Running the Program

1. **Install and run Ollama**
  - Install: https://ollama.com/download
  - Pull a model (example): `ollama pull gpt-oss:20b`
  - Ensure the Ollama service is running (default `http://localhost:11434`).

2. **Prepare folders**
  - Create an **input folder** for your source videos (e.g. `videos/`).
  - Create an **output folder** for your clips (e.g. `output/`).
  - Ensure a `system/` folder exists (or let Python create it on first write).

3. **Activate your virtual environment**
  ```powershell
  cd d:\Prosjekter\AI_Auto_clipper
  .\.venv\Scripts\Activate.ps1
  ```

4. **Run the main script (or use the launcher)**
  ```powershell
  python main.py
  ```

  Or use the provided batch launcher, which sets up venv and installs dependencies:

  ```powershell
  .\AI_clipper.bat
  ```

5. **Follow the interactive setup** (first run)
  - Output folder path (e.g. `d:\Prosjekter\AI_Auto_clipper\output\`).
  - Input folder path (e.g. `d:\Prosjekter\AI_Auto_clipper\videos\`).
  - AI model name for Ollama (e.g., `gpt-oss:20b`).
  - Transcribing model (`tiny`, `base`, `small`, `medium`, `large`).
  - User query (e.g. *"Find all clips about mindset and motivation"*).
  - The setup will test the AI connection and query the model for `max_tokens`.
  - Optional: provide a list of YouTube URLs to download before clipping.
  - Ensure Ollama is running and the chosen model is pulled (e.g., `gpt-oss:20b`).

6. **Subsequent runs**
  - On later runs, you’re asked if you want to skip the booting stage.
  - You can review and edit saved settings (max tokens, folders, model, query, etc.).
  - When boot succeeds, the system starts scanning your input folder and processing videos.
  - If a previous run was interrupted, the system detects the `run_state.json` file and offers to resume or start fresh.

Output clips are written incrementally to the configured output folder.

---

## Processing Pipeline

For each video:
1. **Scan & download**
  - `yt_downloader.py` (optional) downloads any configured YouTube links.
  - `scan_videos.py` enumerates all files in the input folder.

2. **Transcription**
  - `transcribing.py` loads Whisper and transcribes the video.
  - Short, incomplete segments are merged into more coherent chunks.
  - Result is saved as `system/transcribed.json` (list of `[start, end, text]`).

3. **Chunking**
  - `chunking.py` uses an approximate token counter to split the transcript into chunks based on the LLM’s max tokens (with a safety reserve for responses).

4. **AI scanning**
  - `ai_scanning.py` calls the LLM with a system prompt and your **user query**.
  - The model receives one chunk (or the full transcript) as JSON and must respond with a pure JSON list of `[start, end]` pairs.
  - All responses are collected into `system/AI.json`.

5. **Merge & clip**
  - `merge_segments.py` flattens and merges nearby `[start, end]` segments using a tolerance window (30 seconds by default, configurable in settings).
  - `extract_clip.py` uses MoviePy to cut each `[start, end]` region into an MP4 clip in the output folder.

6. **Optional accuracy testing**
  - If enabled, each clip is re‑transcribed using a chosen Whisper model.
  - Very short or obviously cut‑off endings are detected and the clip is rebuilt with a slightly earlier end time.

7. **Cleanup & logging**
  - Intermediate JSONs (`system/transcribed.json`, `system/AI.json`, `system/clips.json`) and the processed source video are deleted.
  - A log of each run is written to `system/log.txt`.

---

## Tips & Customization
- **Whisper model**: use `tiny`/`base` for speed, `small`/`medium`/`large` for higher quality.
- **Max tokens**: the setup automatically queries your model, but you can override it in the boot menu if needed.
- **Merge tolerance**: the segment merge tolerance (30 seconds) is set inside `merge_segments.py`; lowering it creates more but shorter clips.
- **LLM behavior**: you can tweak the system prompt in `ai_scanning.py` if your model needs more or less strictness.

---

## Troubleshooting
- **No clips created**
  - Check that the user query actually appears in the video content.
  - Verify that the LLM is running and reachable from `setup_stage.py`.

- **Whisper / MoviePy errors**
  - Ensure FFmpeg is installed and visible on `PATH`.
  - Verify `torch` and `openai-whisper` are correctly installed in your virtualenv.

- **LLM returns invalid JSON**
  - `ai_scanning.py` expects strict JSON; some models may need stronger system instructions or more conservative settings.

- **Very slow processing**
  - Use a smaller Whisper model (e.g. `tiny`).
  - Use a lighter LLM or run on GPU.

---

## License
Licensed under the Apache License, Version 2.0. See the `LICENSE` file for the full text. You may not use this project except in compliance with the License.

---

## Disclaimer
This tool depends on transcript and model quality. Always review exported clips before publishing.

Enjoy faster, AI‑guided clip extraction!
