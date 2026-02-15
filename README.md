# AI Auto Clipper

Local-first clipping pipeline with Whisper transcription, Ollama reasoning models, and a production terminal UX. OpusClip but free

## Highlights
- One-command startup (`run.bat` or `run.ps1`)
- First-run setup wizard with hardware detection and model auto-pull
- Rich terminal dashboard with status and settings
- Live Activity Panel with AI clipping and extraction counters
- Robust yt-dlp format fallback (`2K -> 1080p -> progressive`)
- AI clipping pipeline (scan -> merge -> filter -> extract)
- Strict AI JSON parsing with retries and schema validation
- Thinking-model policy (`>=8b`) enforced in setup + settings
- Temp archive cleanup policy (never / max size / max age)
- Clean modular architecture for public/demo use

## Quick Start (Windows)
1. Install prerequisites:
```powershell
winget install --id Python.Python.3.11 -e
winget install --id Gyan.FFmpeg -e
winget install --id Ollama.Ollama -e
```
2. Clone the repo:
```powershell
git clone https://github.com/wessel05j/AI_Auto_clipper.git
cd AI_Auto_clipper
```
3. Run:
```powershell
.\run.bat
```

No manual model pull is required. The setup wizard handles model recommendation and pull automatically.

## Quick Start (PowerShell)
```powershell
git clone https://github.com/wessel05j/AI_Auto_clipper.git
cd AI_Auto_clipper
.\run.ps1
```

## Runtime Flow
1. Launcher creates/activates `venv` and installs dependencies via `setup_env.py`.
2. App validates `config/config.json` on startup.
3. If config is missing/broken, setup wizard runs automatically.
4. If config is valid, dashboard opens directly.

## Setup Wizard Features
- Rich terminal UI (logo, panels, progress spinners)
- Hardware profiling:
  - CPU
  - RAM
  - GPU and VRAM
  - CUDA/GPU acceleration status
- Intensity selection:
  - `light`
  - `balanced`
  - `maximum`
- Automatic Ollama model recommendation (thinking models prioritized)
- Only thinking models with at least `8b` are accepted
- Automatic `ollama pull` when selected model is missing
- User-selectable output clips directory
- Temp retention policy for processed videos in `temp/`
- Optional system prompt editing with warning
- Configuration output:
  - `config/config.json`
  - `config/hardware_profile.json`
  - `config/model_profile.json`

## Dashboard
After setup, the main dashboard provides:
- `Launch Clipping Engine`
- `Change Settings`
- `Fetch YouTube Links`
- `Re-run Setup Wizard`
- `Edit System Prompt`
- `Exit`
- `View Logs` (filter `all`, `warning`, `error`)

## Project Structure
```text
AI_Auto_clipper/
|-- launcher/
|   |-- run.bat
|   `-- run.ps1
|-- core/
|   |-- engine.py
|   |-- clipping.py
|   |-- ai_pipeline.py
|   |-- yt_handler.py
|   `-- format_checker.py
|-- ui/
|   |-- setup_wizard.py
|   |-- dashboard.py
|   `-- components.py
|-- config/
|   |-- config.json
|   |-- hardware_profile.json
|   |-- model_profile.json
|   `-- .gitkeep
|-- logs/
|   `-- .gitkeep
|-- utils/
|   |-- hardware_detect.py
|   |-- model_selector.py
|   `-- validators.py
`-- main.py
```

## Notes
- Input videos: `input/`
- Exported clips: `output/`
- Processed source archive: `temp/`
- Merged candidates are filtered by duration constraints before final extraction
- Runtime logs: `logs/app.log`

## Compatibility
- Windows-first launcher UX
- Local GPU or CPU inference supported

## License
Apache License 2.0 (`LICENSE`)
