from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.clipping import extract_clips


def require_binary(name: str) -> str:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    raise RuntimeError(f"Required tool not found in PATH: {name}")


def run_command(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "unknown command failure"
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{details}")


def probe_duration(path: Path, ffprobe_bin: str) -> float:
    completed = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_entries",
            "format=duration:stream=duration",
            str(path),
        ],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "unknown ffprobe failure"
        raise RuntimeError(f"ffprobe failed for {path}: {details}")

    payload = json.loads(completed.stdout or "{}")
    durations = []

    format_info = payload.get("format", {})
    if isinstance(format_info, dict):
        try:
            format_duration = float(format_info.get("duration"))
        except (TypeError, ValueError):
            format_duration = 0.0
        if format_duration > 0:
            durations.append(format_duration)

    streams = payload.get("streams", [])
    if isinstance(streams, list):
        for stream in streams:
            if not isinstance(stream, dict):
                continue
            try:
                stream_duration = float(stream.get("duration"))
            except (TypeError, ValueError):
                stream_duration = 0.0
            if stream_duration > 0:
                durations.append(stream_duration)

    if not durations:
        raise RuntimeError(f"No valid duration reported for {path}")
    return max(durations)


def assert_valid_output(path: Path, ffprobe_bin: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Expected output was not created: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Expected output is empty: {path}")
    if probe_duration(path, ffprobe_bin) <= 0:
        raise RuntimeError(f"Expected output has invalid duration: {path}")


def create_h264_source(ffmpeg_bin: str, output_path: Path) -> None:
    run_command(
        [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=1280x720:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000",
            "-t",
            "3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_path),
        ]
    )


def create_incompatible_mkv_source(ffmpeg_bin: str, output_path: Path) -> None:
    run_command(
        [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "3",
            "-c:v",
            "libtheora",
            "-q:v",
            "5",
            "-c:a",
            "libvorbis",
            "-q:a",
            "4",
            str(output_path),
        ]
    )


def main() -> int:
    ffmpeg_bin = require_binary("ffmpeg")
    ffprobe_bin = require_binary("ffprobe")

    temp_root = PROJECT_ROOT / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("verify_fast_trim")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    with tempfile.TemporaryDirectory(prefix="verify_fast_trim.", dir=str(temp_root)) as temp_dir:
        work_dir = Path(temp_dir)
        input_dir = work_dir / "input"
        output_dir = work_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        fast_source = input_dir / "copy_source.mp4"
        fallback_source = input_dir / "fallback_source.mkv"

        create_h264_source(ffmpeg_bin, fast_source)
        create_incompatible_mkv_source(ffmpeg_bin, fallback_source)

        copy_events: list[tuple[int, Path, bool]] = []
        copy_created = extract_clips(
            clips=[[0.5, 1.8, 7]],
            source_video=fast_source,
            output_dir=output_dir,
            logger=logger,
            progress_interval=1,
            on_clip_done=lambda idx, path, created: copy_events.append((idx, Path(path), created)),
        )
        if copy_created != 1:
            raise RuntimeError(f"Expected 1 copied clip, got {copy_created}")
        if len(copy_events) != 1 or not copy_events[0][2]:
            raise RuntimeError("Copy-trim callback contract failed")
        copy_output = copy_events[0][1]
        if copy_output.suffix.lower() != ".mp4":
            raise RuntimeError(f"Expected MP4 copy-trim output, got {copy_output.name}")
        assert_valid_output(copy_output, ffprobe_bin)

        fallback_events: list[tuple[int, Path, bool]] = []
        fallback_created = extract_clips(
            clips=[[0.25, 1.75, 8]],
            source_video=fallback_source,
            output_dir=output_dir,
            logger=logger,
            progress_interval=1,
            on_clip_done=lambda idx, path, created: fallback_events.append((idx, Path(path), created)),
        )
        if fallback_created != 1:
            raise RuntimeError(f"Expected 1 fallback clip, got {fallback_created}")
        if len(fallback_events) != 1 or not fallback_events[0][2]:
            raise RuntimeError("Fallback callback contract failed")
        fallback_output = fallback_events[0][1]
        if fallback_output.suffix.lower() != ".mkv":
            raise RuntimeError(f"Expected MKV fallback output, got {fallback_output.name}")
        assert_valid_output(fallback_output, ffprobe_bin)

        skip_events: list[tuple[int, Path, bool]] = []
        skipped = extract_clips(
            clips=[[0.25, 1.75, 8]],
            source_video=fallback_source,
            output_dir=output_dir,
            logger=logger,
            progress_interval=1,
            skip_existing=True,
            on_clip_done=lambda idx, path, created: skip_events.append((idx, Path(path), created)),
        )
        if skipped != 0:
            raise RuntimeError(f"Expected skip-existing to create 0 clips, got {skipped}")
        if len(skip_events) != 1 or skip_events[0][2]:
            raise RuntimeError("Skip-existing callback contract failed")
        if skip_events[0][1] != fallback_output:
            raise RuntimeError("Skip-existing did not report the actual MKV fallback path")

        print(f"copy-trim ok: {copy_output.name}")
        print(f"container fallback ok: {fallback_output.name}")
        print("skip-existing ok")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
