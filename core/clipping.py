from __future__ import annotations

import contextlib
import io
import json
import logging
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Sequence

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def scan_input_videos(input_dir: Path) -> List[Path]:
    videos: List[Path] = []
    if not input_dir.exists():
        return videos
    for file_path in input_dir.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if file_path.name.endswith(".part"):
            continue
        try:
            videos.append(_sanitize_input_video_file(file_path))
        except Exception:
            videos.append(file_path)
    videos.sort(key=lambda path: path.stat().st_mtime)
    return videos


def transcribe_video(
    video_path: Path,
    whisper_model,
    min_pause: float = 3.0,
    logger: Optional[logging.Logger] = None,
) -> List[List[object]]:
    """
    Transcribe and merge short pause segments into sentence-level transcript chunks.
    Output format: [[start, end, text], ...]
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            result = whisper_model.transcribe(str(video_path), verbose=False)

    if logger is not None:
        for warning in captured_warnings:
            warning_text = str(getattr(warning, "message", "")).strip()
            if not warning_text:
                continue
            warning_lower = warning_text.lower()
            if "unclosed file" in warning_lower:
                continue
            if "FP16 is not supported on CPU" in warning_text:
                logger.info("Whisper is using FP32 on CPU.")
            else:
                logger.warning("Whisper warning: %s", warning_text)
    strong_end = re.compile(r"[\.!\?]+(?:\"|'|\s|$)")

    merged: List[List[object]] = []
    current = None

    for segment in result.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))

        if current is None:
            current = {"start": start, "end": end, "text": text}
            continue

        pause = start - float(current["end"])
        has_sentence_end = bool(strong_end.search(str(current["text"])))

        if has_sentence_end or pause >= min_pause:
            merged.append([float(current["start"]), float(current["end"]), str(current["text"]).strip()])
            current = {"start": start, "end": end, "text": text}
        else:
            current["end"] = end
            current["text"] = f"{current['text']} {text}".strip()

    if current:
        merged.append([float(current["start"]), float(current["end"]), str(current["text"]).strip()])

    return merged


def merge_segments(segment_list: Sequence[Sequence[Sequence[float]]], tolerance_seconds: float) -> List[List[float]]:
    all_blocks: List[List[float]] = []
    for group in segment_list:
        for block in group:
            if len(block) < 2:
                continue
            start = float(block[0])
            end = float(block[1])
            score = float(block[2]) if len(block) > 2 else 5.0
            all_blocks.append([start, end, score])

    if not all_blocks:
        return []

    all_blocks.sort(key=lambda clip: clip[0])
    merged = [all_blocks[0]]

    for start, end, score in all_blocks[1:]:
        previous = merged[-1]
        prev_end = previous[1]
        if start <= prev_end + tolerance_seconds:
            previous[1] = max(prev_end, end)
            previous[2] = max(previous[2], score)
        else:
            merged.append([start, end, score])

    return merged


def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r"[<>:\"/\\|?*']", "_", filename)
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized.strip(". ")


def _sanitize_input_video_file(file_path: Path) -> Path:
    safe_stem = sanitize_filename(file_path.stem)
    safe_name = f"{safe_stem}{file_path.suffix}"
    if safe_name == file_path.name:
        return file_path

    target = file_path.with_name(safe_name)
    if not target.exists():
        file_path.rename(target)
        return target

    counter = 1
    while True:
        candidate = file_path.with_name(f"{safe_stem}_{counter:03d}{file_path.suffix}")
        if not candidate.exists():
            file_path.rename(candidate)
            return candidate
        counter += 1


def _require_binary(name: str) -> str:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    raise RuntimeError(f"Required tool not found in PATH: {name}")


def _format_ffmpeg_time(seconds: float) -> str:
    formatted = f"{max(0.0, seconds):.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _delete_partial_output(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _summarize_error_details(details: str, max_lines: int = 4, max_chars: int = 500) -> str:
    lines = [line.strip() for line in str(details).splitlines() if line.strip()]
    normalized = " ".join(lines[:max_lines])
    if not normalized:
        return "unknown ffmpeg error"
    if len(lines) > max_lines:
        normalized = f"{normalized} ..."
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def _probe_duration_seconds(path: Path, ffprobe_bin: str, *, strict: bool = False) -> Optional[float]:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format=duration:stream=duration",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required for clip validation") from exc

    if completed.returncode != 0:
        if strict:
            details = completed.stderr.strip() or completed.stdout.strip() or "unknown ffprobe error"
            raise RuntimeError(f"ffprobe failed for {path}: {details}")
        return None

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        if strict:
            raise RuntimeError(f"ffprobe returned invalid JSON for {path}")
        return None

    duration_candidates: List[float] = []

    format_info = payload.get("format", {})
    if isinstance(format_info, dict):
        try:
            format_duration = float(format_info.get("duration"))
        except (TypeError, ValueError):
            format_duration = 0.0
        if format_duration > 0:
            duration_candidates.append(format_duration)

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
                duration_candidates.append(stream_duration)

    if duration_candidates:
        return max(duration_candidates)
    if strict:
        raise RuntimeError(f"Unable to determine media duration for {path}")
    return None


def _validate_media_output(path: Path, ffprobe_bin: str) -> tuple[bool, str]:
    if not path.exists():
        return False, "file was not created"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"unable to stat file: {exc}"
    if size <= 0:
        return False, "file is empty"

    duration = _probe_duration_seconds(path, ffprobe_bin, strict=False)
    if duration is None or duration <= 0:
        return False, "ffprobe reported invalid duration"
    return True, ""


def _run_ffmpeg_command(command: Sequence[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for clip extraction") from exc

    if completed.returncode == 0:
        return True, ""

    details = completed.stderr.strip() or completed.stdout.strip() or f"ffmpeg exited with code {completed.returncode}"
    return False, details


def _copy_trim_command(ffmpeg_bin: str, source_video: Path, output_path: Path, start: float, duration: float) -> List[str]:
    return [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-ss",
        _format_ffmpeg_time(start),
        "-i",
        str(source_video),
        "-t",
        _format_ffmpeg_time(duration),
        "-map",
        "0:v:0?",
        "-map",
        "0:a?",
        "-sn",
        "-dn",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]


def _reencode_trim_command(ffmpeg_bin: str, source_video: Path, output_path: Path, start: float, duration: float) -> List[str]:
    return [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(source_video),
        "-ss",
        _format_ffmpeg_time(start),
        "-t",
        _format_ffmpeg_time(duration),
        "-map",
        "0:v:0?",
        "-map",
        "0:a?",
        "-sn",
        "-dn",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _existing_output_candidates(mp4_output: Path, mkv_output: Path, exact_trim_reencode: bool) -> List[Path]:
    if exact_trim_reencode:
        return [mp4_output]
    return [mp4_output, mkv_output]


def _find_existing_output(
    candidates: Sequence[Path],
    ffprobe_bin: str,
    logger: logging.Logger,
) -> Optional[Path]:
    for candidate in candidates:
        if not candidate.exists():
            continue
        valid, reason = _validate_media_output(candidate, ffprobe_bin)
        if valid:
            return candidate
        logger.warning("Removing invalid existing clip %s: %s", candidate.name, reason)
        _delete_partial_output(candidate)
    return None


def _extract_clip_stream_copy(
    *,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    source_video: Path,
    mp4_output: Path,
    mkv_output: Path,
    start: float,
    duration: float,
    logger: logging.Logger,
) -> Path:
    errors: List[str] = []

    attempts = [
        (mp4_output, _copy_trim_command(ffmpeg_bin, source_video, mp4_output, start, duration)),
        (mkv_output, _copy_trim_command(ffmpeg_bin, source_video, mkv_output, start, duration)),
    ]

    for attempt_index, (output_path, command) in enumerate(attempts, start=1):
        _delete_partial_output(output_path)
        succeeded, details = _run_ffmpeg_command(command)
        valid, reason = _validate_media_output(output_path, ffprobe_bin)
        if valid:
            if not succeeded:
                logger.warning(
                    "FFmpeg reported errors while trimming %s to %s, but the output passed validation. Details: %s",
                    source_video.name,
                    output_path.name,
                    _summarize_error_details(details),
                )
            if output_path.suffix.lower() == ".mkv":
                logger.warning(
                    "MP4 stream-copy trim was incompatible for %s; kept clip as %s",
                    source_video.name,
                    output_path.name,
                )
            return output_path
        if succeeded:
            details = reason

        _delete_partial_output(output_path)
        errors.append(f"{output_path.suffix.lower()} copy trim failed: {details}")

        if attempt_index == 1:
            logger.warning(
                "MP4 stream-copy trim failed for %s; retrying Matroska container. Details: %s",
                source_video.name,
                _summarize_error_details(details),
            )

    raise RuntimeError("; ".join(errors))


def _extract_clip_reencode(
    *,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    source_video: Path,
    output_path: Path,
    start: float,
    duration: float,
    logger: logging.Logger,
) -> Path:
    _delete_partial_output(output_path)
    succeeded, details = _run_ffmpeg_command(
        _reencode_trim_command(ffmpeg_bin, source_video, output_path, start, duration)
    )
    valid, reason = _validate_media_output(output_path, ffprobe_bin)
    if valid:
        if not succeeded:
            logger.warning(
                "FFmpeg reported errors while exact-trimming %s to %s, but the output passed validation. Details: %s",
                source_video.name,
                output_path.name,
                _summarize_error_details(details),
            )
        return output_path

    _delete_partial_output(output_path)
    if not succeeded:
        raise RuntimeError(details)
    raise RuntimeError(reason)


def extract_clips(
    clips: Sequence[Sequence[float]],
    source_video: Path,
    output_dir: Path,
    logger: logging.Logger,
    progress_interval: int = 5,
    clip_name_indices: Optional[Sequence[int]] = None,
    skip_existing: bool = False,
    on_clip_done: Optional[Callable[[int, str, bool], None]] = None,
    exact_trim_reencode: bool = False,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_count = 0
    if clip_name_indices is not None and len(clip_name_indices) != len(clips):
        raise ValueError("clip_name_indices length must match clips length")

    ffmpeg_bin = _require_binary("ffmpeg")
    ffprobe_bin = _require_binary("ffprobe")
    source_duration = _probe_duration_seconds(source_video, ffprobe_bin, strict=True)
    if source_duration is None or source_duration <= 0:
        raise RuntimeError(f"Video has invalid duration: {source_video}")

    base_name = sanitize_filename(source_video.stem)
    for index, clip in enumerate(clips, start=1):
        clip_idx = int(clip_name_indices[index - 1]) if clip_name_indices is not None else index
        if len(clip) < 2:
            continue
        start = max(0.0, float(clip[0]))
        end = min(source_duration, float(clip[1]))
        if end <= start:
            continue

        score = float(clip[2]) if len(clip) > 2 else 5.0
        clip_stem = f"{base_name}_{clip_idx:03d}_r{int(round(score))}"
        mp4_output = output_dir / f"{clip_stem}.mp4"
        mkv_output = output_dir / f"{clip_stem}.mkv"

        if skip_existing:
            existing_output = _find_existing_output(
                _existing_output_candidates(mp4_output, mkv_output, exact_trim_reencode),
                ffprobe_bin,
                logger,
            )
            if existing_output is not None:
                if on_clip_done is not None:
                    on_clip_done(clip_idx, str(existing_output), False)
                continue

        duration = end - start
        if exact_trim_reencode:
            created_output = _extract_clip_reencode(
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
                source_video=source_video,
                output_path=mp4_output,
                start=start,
                duration=duration,
                logger=logger,
            )
        else:
            created_output = _extract_clip_stream_copy(
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
                source_video=source_video,
                mp4_output=mp4_output,
                mkv_output=mkv_output,
                start=start,
                duration=duration,
                logger=logger,
            )

        clip_count += 1
        if on_clip_done is not None:
            on_clip_done(clip_idx, str(created_output), True)
        if clip_count % max(1, progress_interval) == 0:
            logger.info("Extracted %s clips from %s", clip_count, source_video.name)

    return clip_count
