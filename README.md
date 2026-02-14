# 🎬 AI Auto Clipper

Turn long videos into short, query-matched clips with a local-first pipeline.
AI Auto Clipper transcribes with Whisper, scans transcript chunks with Ollama, merges nearby matches, and exports MP4 clips.

## ✨ What You Get
- 🧠 Local transcription with Whisper (`tiny` to `large`)
- 🤖 Local semantic clip selection with Ollama models
- 📥 Optional YouTube ingest (direct links + channel monitoring)
- 🔁 Resume-friendly temp/progress files
- ⚡ Automatic `torch` setup with CUDA detection + CPU fallback

## 🚀 Fast Start (Windows)

### 1) Install prerequisites (PowerShell as Admin)
```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.11 -e
winget install --id Gyan.FFmpeg -e
winget install --id Ollama.Ollama -e
```

### 2) Verify prerequisites
```powershell
git --version
python --version
ffmpeg -version
ollama --version
```

### 3) Clone project
```powershell
git clone https://github.com/wessel05j/AI_Auto_clipper.git
cd AI_Auto_clipper
```

### 4) Pull an Ollama model
```powershell
ollama pull gpt-oss:20b
```

### 5) Run setup wizard
```powershell
.\settings.bat
```

### 6) Add videos and run
```powershell
.\main.bat
```

Input videos go in `input/` and generated clips appear in `output/`.

## 🐧🍎 Fast Start (macOS / Linux)

### 1) Install prerequisites

macOS (Homebrew):
```bash
brew install git python ffmpeg ollama
```

Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y git python3 python3-venv ffmpeg curl
curl -fsSL https://ollama.com/install.sh | sh
```

### 2) Verify prerequisites
```bash
git --version
python3 --version
ffmpeg -version
ollama --version
```

### 3) Clone + enter project
```bash
git clone https://github.com/wessel05j/AI_Auto_clipper.git
cd AI_Auto_clipper
```

### 4) Pull an Ollama model
```bash
ollama pull llama3.2
```

### 5) Create env + install deps (CUDA/CPU auto)
```bash
python3 -m venv venv
source venv/bin/activate
python setup_env.py --torch auto
```

### 6) Configure + run
```bash
python settings.py
python main.py
```
=======
# 🎬 AI Auto Clipper

**Like OpusClip, but free!** 🚀 An automatic AI-powered video clip extractor that discovers videos, transcribes them with Whisper, uses local LLMs via Ollama to find interesting segments based on your query, and exports matching clips as MP4 files. No code editing needed – just run and configure interactively!

## ✨ How It Works
1. 🔍 Discover videos from local folders or download from YouTube
2. 🎙️ Transcribe with Whisper (local, fast, private)
3. ✂️ Chunk transcripts smartly using token estimation
4. 🤖 Send chunks to your local LLM (Ollama) to find relevant segments
5. 🔗 Merge nearby clips to avoid fragmentation
6. 🎞️ Render and export clips using MoviePy

---

## 📥 Installation & Setup

### Windows 🪟
1. **Install Python 3.10–3.11**: Download from [python.org](https://www.python.org/downloads/)
2. **Install FFmpeg**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
3. **Install Ollama**: Get it from [ollama.com](https://ollama.com/download) and pull a model: `ollama pull gpt-oss:20b` Suggested using a thinking model
4. **Clone/Download this repo**

### macOS 🍎 / Linux 🐧
1. **Install Python 3.10–3.11**: Use Homebrew (`brew install python`) or your package manager
2. **Install FFmpeg**: `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Ubuntu)
3. **Install Ollama**: Follow [ollama.com](https://ollama.com/download) and run `ollama pull llama3.2`
4. **Clone/Download this repo**

### Python Packages 📦
Install via `pip install -r requirements.txt`:
- `openai-whisper` – Local Whisper transcription
- `ollama` – Ollama client for local LLMs
- `moviepy==1.0.3` – Video editing (pinned for compatibility)
- `yt_dlp` – YouTube downloading
- `torch` – PyTorch (CUDA-enabled if available)
- `requests` – HTTP handling
- `tiktoken` – Token counting
- `tqdm` – Progress bars
- `numpy<2` – Numerical ops (pinned)

---

## 🚀 How to Use

### First Time Setup
1. Run `python settings.py` (or `settings.bat` on Windows)
2. Follow the interactive wizard:
   - Choose your Ollama model (e.g., `gpt-oss:20b`)
   - Pick Whisper model (`tiny` for speed, `large` for accuracy)
   - Enter your clip query (e.g., "Find funny moments")
   - Set max tokens, merge distance, etc.
3. Settings save to `system/settings.json`

### Running the Clipper
- Run `python main.py` (or `main.bat`)
- It processes videos in `input/`, outputs clips to `output/`
- Check `system/log.txt` for logs and progress (good for troubleshooting)

### Adjusting Settings for Better Results 🎛️
Run `python settings.py` anytime to tweak:
- **AI Model**: Use train-of-thought models for better detection (gpt-oss:20b)
- **Whisper Model**: `tiny/base` for speed, `medium/large` for accuracy
- **Max Tokens**: Higher = longer chunks processed at once (better context)
- **Merge Distance**: Seconds to merge nearby clips (e.g., 30+ for longer clips)
- **AI Loops**: How many times to re-scan chunks (1-3 recommended)
- **Temperature**: 0.1-0.9 for creativity vs. precision
- **Channels/Hours Limit**: For YouTube monitoring

---

## 📺 YouTube Downloads & Cookies 🍪

To download unlimited YouTube videos without restrictions:

1. **Install browser extension**: Get "Get cookies.txt" for [Chrome](https://chrome.google.com/webstore/detail/get-cookiestxt/bgaddhkoddajcdgocldbbfleckgcbcid) or [Firefox](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)
2. **Log into YouTube** in your browser
3. **Export cookies**: Click the extension icon, save as `cookies.txt` in the project root
4. **The downloader auto-uses it** – no extra config needed!

If downloads fail, ensure cookies are fresh and from the same browser.

---

## ⚡ Running with CUDA (GPU Acceleration)

For blazing-fast transcription on NVIDIA GPUs:

### Check CUDA Support
After installing deps: `python -c "import torch; print(torch.cuda.is_available())"`
- `True` = GPU mode active! 🚀
- `False` = CPU mode (still works, just slower)

### Manual CUDA Setup
If not detected:
1. **Install CUDA Toolkit 11.8**: From [NVIDIA](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. **Update NVIDIA drivers**: [Download](https://www.nvidia.com/Download/index.aspx)
3. **Reinstall PyTorch**: 
   ```bash
   pip uninstall torch -y
   pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

4. **Restart & verify**

**Note**: Whisper needs 2-8GB VRAM depending on model size. CPU fallback works but is 5-10x slower.

---

## 🗂️ Project Layout
```

Input videos go in `input/` and generated clips appear in `output/`.

## ⚡ CUDA / GPU Notes
- `setup_env.py --torch auto` tries CUDA wheels first when NVIDIA tooling is detected.
- If CUDA is unavailable, it falls back to CPU torch automatically.
- Check status:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Optional modes:
```bash
python setup_env.py --torch cuda
python setup_env.py --torch cpu
python setup_env.py --torch skip
python setup_env.py --torch auto --force
```

## 🧩 Daily Commands
- Windows configure: `.\settings.bat`
- Windows run: `.\main.bat`
- Windows fetch channel links: `.\fetch_yt_links.bat`
- macOS/Linux configure: `python settings.py`
- macOS/Linux run: `python main.py`
- macOS/Linux fetch channel links: `python fetch_yt_links.py`

## 🛠️ Configuration
All settings are saved in `system/settings.json`.

Important keys:
- `ai_model`
- `transcribing_model`
- `user_query`
- `system_query` (default system prompt is already included automatically)
- `total_tokens`
- `max_chunking_tokens`
- `max_ai_tokens`
- `merge_distance`
- `ai_loops`
- `temperature`
- `rerun_temp_files`

## 📺 YouTube Intake

### Add direct links
Use `settings.py` option `6` to add to `youtube_list`.

### Monitor channels
1. Add channel `/videos` URLs in option `11`
2. Set hour window in option `12`
3. Run `fetch_yt_links.py` (or `fetch_yt_links.bat`)

### Cookies (if YouTube blocks downloads)
Put `cookies.txt` in one of:
- `resources/cookies.txt`
- `system/cookies.txt`
- project root `cookies.txt`

## 🧭 Workflow
1. Scan files in `input/`
2. Transcribe with Whisper
3. Chunk transcript to token budget
4. Ask Ollama for `[start, end, score]` clip candidates
5. Merge nearby segments
6. Export clips to `output/`
7. Move processed source video to `temp/`

## 🆘 Troubleshooting
- `Settings file not found`: run `settings.py` once first
- `ffmpeg` errors: install FFmpeg and verify `ffmpeg -version`
- No clips found: tighten `user_query`, increase `ai_loops`, lower `temperature`
- Ollama errors: verify `ollama serve` and `ollama_url` in settings
- Slow runtime: use `tiny`/`base` Whisper model or ensure CUDA is active
- Download issues: refresh `cookies.txt` and retry

## 📁 Project Layout
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

## 📄 License
Apache License 2.0 (`LICENSE`).
