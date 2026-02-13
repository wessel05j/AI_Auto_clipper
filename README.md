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
   ```
4. **Restart & verify**

**Note**: Whisper needs 2-8GB VRAM depending on model size. CPU fallback works but is 5-10x slower.

---

## 🗂️ Project Layout
```
AI_Auto_clipper/
├── main.py                     # Main orchestrator
├── settings.py                 # Interactive config wizard
├── programs/
│   ├── components/             # Helpers (JSON, tokens, file ops)
│   └── core_functionality/     # Core modules (transcribe, scan, extract)
├── system/                     # Configs & logs (settings.json, log.txt)
├── input/                      # Your video files
├── output/                     # Generated clips
└── temp/                       # Temporary processing files
```

---

## 🔧 System Usage & Logging 📝

- **Logging**: All activity logged to `system/log.txt` – check for errors or progress
- **Status Files**: `system/status.json` tracks current processing state
- **Temp Files**: Safe to delete `temp/` after runs, but keep for reruns
- **Rerun Option**: In settings, enable "Rerun Temp Files" to skip re-transcription

---

## 🛠️ Troubleshooting

- **No clips found?** Adjust your query or lower temperature
- **Transcription slow?** Use smaller Whisper model or enable CUDA
- **YouTube download fails?** Check cookies or try without them
- **Ollama errors?** Ensure Ollama is running: `ollama serve`
- **Memory issues?** Reduce max tokens or use CPU mode
- **Video errors?** Ensure FFmpeg is installed and on PATH

For more help, check `system/log.txt` or open an issue.

---

## 💡 Recommendations

- **Models**: Start with `llama3.2` for AI, `base` for Whisper
- **Hardware**: 8GB+ RAM, GPU recommended for speed
- **Videos**: MP4 format preferred, <2GB for faster processing
- **Queries**: Be specific (e.g., "funny cat videos" vs. "cats")

---

## 📄 License
Apache License 2.0
