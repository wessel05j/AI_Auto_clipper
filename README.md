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
├── main.py                     # Orchestrates setup + full pipeline
├── settings.py                 # Interactive settings menu
├── programs/
│   ├── components/
│   │   ├── file_exists.py      # Tiny helper to check JSON/clip files
│   │   ├── interact_w_json.py  # Read/write JSON helper
│   │   ├── load.py             # JSON load wrapper
│   │   ├── return_tokens.py    # Token estimation using tiktoken
│   │   ├── scan_videos.py      # Collects video file paths from input folder
│   │   └── write.py            # JSON write wrapper
│   └── core_functionality/
│       ├── yt_downloader.py    # Downloads YouTube videos to input folder
│       ├── transcribing.py     # Whisper transcription + basic segment merging
│       ├── chunking.py         # Splits transcript into token-sized chunks
│       ├── ollama_scanning.py  # LLM call to extract relevant [start, end] spans
│       ├── merge_segments.py   # Merges close timestamp segments
│       ├── extract_clip.py     # Cuts clips using MoviePy
│       ├── ollama_on.py        # Starts Ollama server
│       ├── ollama_off.py       # Stops Ollama server
│       ├── ollama_chat.py      # Chat with Ollama
│       └── ollama_scanning.py  # Scanning with Ollama
├── system/                     # Settings + temporary JSON artifacts (settings.json, transcribing.json, AI.json, clips.json)
├── input/                      # Input videos
├── output/                     # Generated clips
└── temp/                       # Temp files
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
- `torch` – PyTorch (automatically installs CUDA version if CUDA is available).
- `requests` – HTTP requests.
- `tiktoken` – Token estimation.
- `tqdm` – Progress bars.
- `numpy<2` – NumPy (pinned for compatibility).

---

## CUDA Support

PyTorch (required by Whisper) supports GPU acceleration via CUDA for faster transcription.

### Automatic Detection
- The `requirements.txt` installs PyTorch with CUDA support if compatible drivers are detected.
- If CUDA is not available on your system, it falls back to CPU-only PyTorch (slower but functional).
- To check: After installation, run `python -c "import torch; print(torch.cuda.is_available())"`
  - If `False`, you're using CPU mode.

### Manual CUDA Installation
If automatic detection fails or you installed CUDA after setting up the environment:
1. Activate the virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
2. Uninstall CPU PyTorch: `pip uninstall torch -y`
3. Install CUDA PyTorch: `pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118`
4. Verify: `python -c "import torch; print(torch.cuda.is_available())"` should return `True`

### Installing CUDA for GPU Acceleration
If you have an NVIDIA GPU and want faster transcription:
1. Check your GPU's CUDA compatibility: https://developer.nvidia.com/cuda-gpus
2. Download and install CUDA Toolkit 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
3. Install NVIDIA drivers (if not already): https://www.nvidia.com/Download/index.aspx
4. Restart your system.
5. Reinstall dependencies: Delete the `venv` folder and run `main.bat` again.

### Troubleshooting
- If CUDA is not detected after installation, PyTorch falls back to CPU.
- For Windows, ensure NVIDIA drivers are up to date.
- Check GPU memory: Whisper models require 2-8GB VRAM depending on size.
- CPU mode works but is 5-10x slower than GPU.

---

## Running the Program

### Quick Start (Windows)
1. Ensure Python 3.10–3.11 is installed.
2. Install FFmpeg and Ollama as described in Requirements.
3. Run `main.bat` – it will create a virtual environment, install dependencies, and start the application.

### Manual Setup
1. **Install Dependencies**
   - Install FFmpeg and Ollama as above.
   - Run `pip install -r requirements.txt`

2. **First Run**
   - Run `python settings.py`
   - Follow interactive setup: choose model, query, etc.
   - Settings saved to `system/settings.json`

3. **Subsequent Runs**
   - Run `python settings.py` to edit settings.
   - Run `python main.py` to process videos.

Output clips in `output/`.

---

## Tips
- Use smaller Whisper models for speed.
- Adjust merge distance for clip length.

---

## License
Apache License 2.0
