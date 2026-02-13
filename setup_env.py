#!/usr/bin/env python
"""
Environment bootstrapper for AI Auto Clipper.

What it does:
1) Installs/updates Python dependencies from requirements.txt
2) Installs torch in CPU or CUDA mode (auto-detected by default)
3) Saves setup state so repeated launches can skip reinstall work
4) Warns if ffmpeg or ollama CLI is missing
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


BASE_DIR = Path(__file__).resolve().parent
SYSTEM_DIR = BASE_DIR / "system"
STATE_FILE = SYSTEM_DIR / "setup_state.json"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
CUDA_INDEX_URLS = (
    "https://download.pytorch.org/whl/cu124",
    "https://download.pytorch.org/whl/cu121",
    "https://download.pytorch.org/whl/cu118",
)


def run_cmd(args: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def pip_install(args: list[str]) -> None:
    run_cmd([sys.executable, "-m", "pip", *args])


def read_setup_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_setup_state(state: Dict[str, Any]) -> None:
    SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def requirements_hash() -> str:
    payload = REQUIREMENTS_FILE.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def running_in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def detect_nvidia_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = run_cmd(["nvidia-smi", "-L"], capture_output=True)
    except Exception:
        return False
    return bool(result.stdout.strip())


def probe_torch() -> Dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False}
    try:
        import torch  # type: ignore

        return {
            "installed": True,
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        }
    except Exception as exc:  # pragma: no cover - defensive path for broken envs
        return {"installed": False, "error": str(exc)}


def resolve_torch_target(requested: str, has_nvidia_gpu: bool) -> str:
    if requested == "auto":
        return "cuda" if has_nvidia_gpu else "cpu"
    return requested


def torch_satisfies_target(torch_info: Dict[str, Any], target: str) -> bool:
    if target == "skip":
        return True
    if not torch_info.get("installed"):
        return False
    if target == "cuda":
        return bool(torch_info.get("cuda_available"))
    # CPU target accepts either CPU-only torch or CUDA-capable torch.
    return True


def install_torch(target: str) -> Tuple[str, Dict[str, Any]]:
    if target == "skip":
        return "skip", probe_torch()

    if target == "cpu":
        print("Installing torch (CPU build)...")
        pip_install(["install", "--upgrade", "torch"])
        return "cpu", probe_torch()

    if target == "cuda":
        for index_url in CUDA_INDEX_URLS:
            print(f"Installing torch CUDA build via {index_url} ...")
            try:
                pip_install(["install", "--upgrade", "torch", "--index-url", index_url])
                torch_info = probe_torch()
                if torch_info.get("cuda_available"):
                    return "cuda", torch_info
            except subprocess.CalledProcessError:
                continue

        print("CUDA torch install failed or CUDA not available. Falling back to CPU torch.")
        pip_install(["install", "--upgrade", "torch"])
        return "cpu", probe_torch()

    raise ValueError(f"Unknown torch target: {target}")


def check_external_tools() -> None:
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ollama_ok = shutil.which("ollama") is not None

    if not ffmpeg_ok:
        print("Warning: ffmpeg was not found in PATH. Video rendering may fail.")
    if not ollama_ok:
        print("Warning: ollama CLI was not found in PATH. Install Ollama to run AI scanning.")


def should_skip_setup(
    state: Dict[str, Any],
    req_hash: str,
    requested_torch: str,
    resolved_torch_target: str,
    torch_info: Dict[str, Any],
    force: bool,
) -> bool:
    if force:
        return False
    if state.get("requirements_hash") != req_hash:
        return False
    if state.get("python_executable") != sys.executable:
        return False
    if state.get("torch_request") != requested_torch:
        return False
    if state.get("torch_target") != resolved_torch_target:
        return False
    return torch_satisfies_target(torch_info, resolved_torch_target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Python environment for AI Auto Clipper.")
    parser.add_argument(
        "--torch",
        choices=("auto", "cuda", "cpu", "skip"),
        default="auto",
        help="Torch installation mode. Default: auto",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinstall of dependencies even if setup state matches.",
    )
    parser.add_argument(
        "--skip-pip-upgrade",
        action="store_true",
        help="Skip pip self-upgrade step.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not REQUIREMENTS_FILE.exists():
        print(f"Error: requirements file not found: {REQUIREMENTS_FILE}")
        return 1

    if not running_in_venv():
        print("Warning: no virtual environment detected. Continuing anyway.")

    has_nvidia_gpu = detect_nvidia_gpu()
    torch_target = resolve_torch_target(args.torch, has_nvidia_gpu)
    req_hash = requirements_hash()
    setup_state = read_setup_state()
    torch_info_before = probe_torch()

    if should_skip_setup(
        setup_state,
        req_hash,
        args.torch,
        torch_target,
        torch_info_before,
        args.force,
    ):
        print("Environment already prepared. Skipping dependency install.")
        check_external_tools()
        return 0

    try:
        if not args.skip_pip_upgrade:
            print("Upgrading pip...")
            pip_install(["install", "--upgrade", "pip"])

        print("Installing project requirements...")
        pip_install(["install", "-r", str(REQUIREMENTS_FILE)])

        resolved_torch_mode, torch_info_after = install_torch(torch_target)
    except subprocess.CalledProcessError as exc:
        print(f"Dependency installation failed. Command: {' '.join(exc.cmd)}")
        return exc.returncode or 1

    setup_payload = {
        "python_executable": sys.executable,
        "requirements_hash": req_hash,
        "torch_request": args.torch,
        "torch_target": torch_target,
        "torch_installed_mode": resolved_torch_mode,
        "torch_version": torch_info_after.get("version"),
        "torch_cuda_available": torch_info_after.get("cuda_available"),
        "torch_cuda_version": torch_info_after.get("cuda_version"),
    }
    write_setup_state(setup_payload)

    print("Environment setup complete.")
    if torch_info_after.get("installed"):
        print(
            f"Torch version: {torch_info_after.get('version')} "
            f"(CUDA available: {torch_info_after.get('cuda_available')})"
        )
    check_external_tools()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
