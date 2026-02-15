from __future__ import annotations

import contextlib
import io
import logging
import re
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

from moviepy.editor import VideoFileClip


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


def extract_clips(
    clips: Sequence[Sequence[float]],
    source_video: Path,
    output_dir: Path,
    logger: logging.Logger,
    progress_interval: int = 5,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_count = 0

    with VideoFileClip(str(source_video)) as main_video:
        duration = float(main_video.duration or 0.0)
        if duration <= 0:
            raise RuntimeError(f"Video has invalid duration: {source_video}")

        base_name = sanitize_filename(source_video.stem)
        for index, clip in enumerate(clips, start=1):
            if len(clip) < 2:
                continue
            start = max(0.0, float(clip[0]))
            end = min(duration, float(clip[1]))
            if end <= start:
                continue

            score = float(clip[2]) if len(clip) > 2 else 5.0
            filename = output_dir / f"{base_name}_{index:03d}_r{int(round(score))}.mp4"
            subclip = main_video.subclip(start, end)
            subclip.write_videofile(
                str(filename),
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=str(output_dir / f"temp_audio_{index:03d}.m4a"),
                remove_temp=True,
                threads=4,
                logger=None,
            )
            clip_count += 1
            if clip_count % max(1, progress_interval) == 0:
                logger.info("Extracted %s clips from %s", clip_count, source_video.name)

    return clip_count
