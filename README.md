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
ollama pull llama3.2
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
