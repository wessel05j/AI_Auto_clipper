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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
STRONG_SENTENCE_END = re.compile(r"[\.!\?]+(?:\"|'|\s|$)")
CONTINUATION_START = re.compile(
    r"^(?:and|but|so|because|then|or|if|when|which|that|like|also|still|plus|well|cause|cuz|anyway|anyways)\b",
    re.IGNORECASE,
)
LOW_CONTENT_SEGMENTS = {
    "yeah",
    "yep",
    "okay",
    "ok",
    "alright",
    "all right",
    "right",
    "thank you",
    "very good",
    "exactly",
    "for sure",
}


def _extract_timed_words(segment: object) -> List[List[object]]:
    if not isinstance(segment, dict):
        return []

    raw_words = segment.get("words", [])
    if not isinstance(raw_words, list):
        return []

    normalized: List[List[object]] = []
    for word in raw_words:
        if not isinstance(word, dict):
            continue
        try:
            start = float(word.get("start"))
            end = float(word.get("end"))
        except (TypeError, ValueError):
            continue

        text = str(word.get("word", ""))
        if not text:
            continue
        if end < start:
            end = start
        normalized.append([start, end, text])

    return normalized


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
    Output format: [[start, end, text], ...] or [[start, end, text, [[w_start, w_end, word], ...]], ...]
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            try:
                result = whisper_model.transcribe(str(video_path), verbose=False, word_timestamps=True)
            except TypeError:
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
    merged: List[List[object]] = []
    current = None

    for segment in result.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        words = _extract_timed_words(segment)

        if current is None:
            current = {
                "start": start,
                "end": end,
                "text": text,
                "words": list(words),
                "words_complete": bool(words),
            }
            continue

        pause = start - float(current["end"])
        has_sentence_end = bool(STRONG_SENTENCE_END.search(str(current["text"])))

        if has_sentence_end or pause >= min_pause:
            item = [
                float(current["start"]),
                float(current["end"]),
                str(current["text"]).strip(),
            ]
            if bool(current.get("words_complete")) and current.get("words"):
                item.append(list(current["words"]))
            merged.append(item)
            current = {
                "start": start,
                "end": end,
                "text": text,
                "words": list(words),
                "words_complete": bool(words),
            }
        else:
            current["end"] = end
            current["text"] = f"{current['text']} {text}".strip()
            if bool(current.get("words_complete")) and words:
                current["words"].extend(words)
            else:
                current["words_complete"] = False
                current["words"] = []

    if current:
        item = [
            float(current["start"]),
            float(current["end"]),
            str(current["text"]).strip(),
        ]
        if bool(current.get("words_complete")) and current.get("words"):
            item.append(list(current["words"]))
        merged.append(item)

    return merged


def _segment_word_bounds(segment: Sequence[Any]) -> Tuple[float, float]:
    start = float(segment[0])
    end = float(segment[1])
    if len(segment) >= 4 and isinstance(segment[3], list) and segment[3]:
        first = next((word for word in segment[3] if isinstance(word, (list, tuple)) and len(word) >= 2), None)
        last = next(
            (
                word
                for word in reversed(segment[3])
                if isinstance(word, (list, tuple)) and len(word) >= 2
            ),
            None,
        )
        try:
            if first is not None:
                start = float(first[0])
            if last is not None:
                end = float(last[1])
        except (TypeError, ValueError):
            pass
    if end < start:
        end = start
    return start, end


def _normalize_transcript_segments(transcript: Sequence[Sequence[Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for index, segment in enumerate(transcript):
        if len(segment) < 3:
            continue
        try:
            start = float(segment[0])
            end = float(segment[1])
        except (TypeError, ValueError):
            continue
        boundary_start, boundary_end = _segment_word_bounds(segment)
        text = str(segment[2]).strip()
        word_count = len(text.split())
        normalized.append(
            {
                "index": index,
                "raw": list(segment),
                "start": start,
                "end": end,
                "boundary_start": boundary_start,
                "boundary_end": boundary_end,
                "duration": max(0.0, end - start),
                "text": text,
                "word_count": word_count,
            }
        )
    return normalized


def _segment_text_is_fragment(segment: Dict[str, Any]) -> bool:
    text = str(segment.get("text", "")).strip()
    if not text:
        return True
    normalized = re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()
    word_count = int(segment.get("word_count", 0))
    duration = float(segment.get("duration", 0.0))
    if normalized in LOW_CONTENT_SEGMENTS:
        return True
    if word_count <= 2:
        return True
    if duration < 1.4 and word_count <= 4:
        return True
    return False


def _starts_with_continuation(text: str) -> bool:
    return bool(CONTINUATION_START.search(str(text or "").strip()))


def _ends_with_sentence(text: str) -> bool:
    return bool(STRONG_SENTENCE_END.search(str(text or "").strip()))


def _adjacent_gap_seconds(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    return max(0.0, float(right["start"]) - float(left["end"]))


def _segments_are_contiguous(
    transcript: Sequence[Dict[str, Any]],
    left_index: int,
    right_index: int,
    *,
    pause_limit_seconds: float,
) -> bool:
    if right_index <= left_index:
        return True
    for index in range(left_index + 1, right_index + 1):
        gap = _adjacent_gap_seconds(transcript[index - 1], transcript[index])
        if gap > pause_limit_seconds:
            return False
    return True


def _candidate_segment_range(
    transcript: Sequence[Dict[str, Any]],
    start_time: float,
    end_time: float,
) -> Optional[Tuple[int, int]]:
    overlapping: List[int] = []
    for index, segment in enumerate(transcript):
        if float(segment["end"]) <= start_time + 1e-6:
            continue
        if float(segment["start"]) >= end_time - 1e-6:
            if overlapping:
                break
            continue
        overlapping.append(index)

    if overlapping:
        return overlapping[0], overlapping[-1]

    midpoint = max(0.0, (float(start_time) + float(end_time)) / 2.0)
    nearest_index: Optional[int] = None
    nearest_distance: Optional[float] = None
    for index, segment in enumerate(transcript):
        center = (float(segment["start"]) + float(segment["end"])) / 2.0
        distance = abs(center - midpoint)
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_index = index
    if nearest_index is None:
        return None
    return nearest_index, nearest_index


def _normalize_anchor_candidates(
    transcript: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        try:
            start = float(candidate.get("start"))
            end = float(candidate.get("end"))
            score = float(candidate.get("score", 5.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        segment_range = _candidate_segment_range(transcript, start, end)
        if segment_range is None:
            continue
        start_index, end_index = segment_range
        normalized.append(
            {
                **candidate,
                "start": start,
                "end": end,
                "score": score,
                "start_index": start_index,
                "end_index": end_index,
            }
        )

    normalized.sort(key=lambda item: (float(item["start"]), float(item["end"]), -float(item["score"])))
    return normalized


def _candidate_can_join_cluster(
    transcript: Sequence[Dict[str, Any]],
    cluster: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    anchor_gap_limit_seconds: float,
    pause_limit_seconds: float,
) -> bool:
    cluster_end_time = float(cluster["anchor_end"])
    candidate_start_time = float(candidate["start"])
    if candidate_start_time <= cluster_end_time + 1e-6:
        return True
    if (candidate_start_time - cluster_end_time) > anchor_gap_limit_seconds:
        return False
    return _segments_are_contiguous(
        transcript,
        int(cluster["anchor_end_index"]),
        int(candidate["start_index"]),
        pause_limit_seconds=pause_limit_seconds,
    )


def _build_anchor_clusters(
    transcript: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    *,
    merge_distance_seconds: float,
) -> List[Dict[str, Any]]:
    normalized_candidates = _normalize_anchor_candidates(transcript, candidates)
    if not normalized_candidates:
        return []

    anchor_gap_limit_seconds = min(max(1.0, float(merge_distance_seconds) * 0.2), 4.0)
    pause_limit_seconds = min(1.25, max(0.75, anchor_gap_limit_seconds))

    clusters: List[Dict[str, Any]] = []
    for candidate in normalized_candidates:
        if not clusters:
            clusters.append(
                {
                    "anchor_start": float(candidate["start"]),
                    "anchor_end": float(candidate["end"]),
                    "anchor_start_index": int(candidate["start_index"]),
                    "anchor_end_index": int(candidate["end_index"]),
                    "score": float(candidate["score"]),
                    "sources": [dict(candidate)],
                }
            )
            continue

        current_cluster = clusters[-1]
        if _candidate_can_join_cluster(
            transcript,
            current_cluster,
            candidate,
            anchor_gap_limit_seconds=anchor_gap_limit_seconds,
            pause_limit_seconds=pause_limit_seconds,
        ):
            current_cluster["anchor_end"] = max(float(current_cluster["anchor_end"]), float(candidate["end"]))
            current_cluster["anchor_end_index"] = max(
                int(current_cluster["anchor_end_index"]),
                int(candidate["end_index"]),
            )
            current_cluster["score"] = max(float(current_cluster["score"]), float(candidate["score"]))
            current_cluster["sources"].append(dict(candidate))
        else:
            clusters.append(
                {
                    "anchor_start": float(candidate["start"]),
                    "anchor_end": float(candidate["end"]),
                    "anchor_start_index": int(candidate["start_index"]),
                    "anchor_end_index": int(candidate["end_index"]),
                    "score": float(candidate["score"]),
                    "sources": [dict(candidate)],
                }
            )
    return clusters


def _clip_duration_from_segments(
    transcript: Sequence[Dict[str, Any]],
    start_index: int,
    end_index: int,
) -> float:
    start = float(transcript[start_index]["boundary_start"])
    end = float(transcript[end_index]["boundary_end"])
    return max(0.0, end - start)


def _should_expand_left(
    transcript: Sequence[Dict[str, Any]],
    start_index: int,
    end_index: int,
    *,
    min_duration_seconds: float,
    pause_limit_seconds: float,
) -> bool:
    if start_index <= 0:
        return False
    previous = transcript[start_index - 1]
    current = transcript[start_index]
    if _adjacent_gap_seconds(previous, current) > pause_limit_seconds:
        return False

    current_duration = _clip_duration_from_segments(transcript, start_index, end_index)
    if _segment_text_is_fragment(current) or _starts_with_continuation(str(current["text"])):
        return True
    if not _ends_with_sentence(str(previous["text"])):
        return True
    if min_duration_seconds > 0 and current_duration < min_duration_seconds:
        return True
    return False


def _should_expand_right(
    transcript: Sequence[Dict[str, Any]],
    start_index: int,
    end_index: int,
    *,
    min_duration_seconds: float,
    pause_limit_seconds: float,
) -> bool:
    if end_index >= len(transcript) - 1:
        return False
    current = transcript[end_index]
    following = transcript[end_index + 1]
    if _adjacent_gap_seconds(current, following) > pause_limit_seconds:
        return False

    current_duration = _clip_duration_from_segments(transcript, start_index, end_index)
    if _segment_text_is_fragment(current) or not _ends_with_sentence(str(current["text"])):
        return True
    if min_duration_seconds > 0 and current_duration < min_duration_seconds:
        return True
    return False


def _expand_cluster_to_transcript_boundaries(
    transcript: Sequence[Dict[str, Any]],
    cluster: Dict[str, Any],
    *,
    min_duration_seconds: float,
    pause_limit_seconds: float,
) -> Tuple[int, int]:
    start_index = int(cluster["anchor_start_index"])
    end_index = int(cluster["anchor_end_index"])
    max_expansions = max(8, int(min_duration_seconds / 6.0) + 4 if min_duration_seconds > 0 else 8)
    expansions = 0

    while expansions < max_expansions:
        changed = False
        if _should_expand_left(
            transcript,
            start_index,
            end_index,
            min_duration_seconds=min_duration_seconds,
            pause_limit_seconds=pause_limit_seconds,
        ):
            start_index -= 1
            expansions += 1
            changed = True
        if _should_expand_right(
            transcript,
            start_index,
            end_index,
            min_duration_seconds=min_duration_seconds,
            pause_limit_seconds=pause_limit_seconds,
        ):
            end_index += 1
            expansions += 1
            changed = True
        if not changed:
            break
    return start_index, end_index


def assemble_anchor_clips(
    transcript: Sequence[Sequence[Any]],
    candidates: Sequence[Dict[str, Any]],
    *,
    merge_distance_seconds: float,
    min_duration_seconds: float = 0.0,
) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    normalized_transcript = _normalize_transcript_segments(transcript)
    if not normalized_transcript:
        return [], []

    clusters = _build_anchor_clusters(
        normalized_transcript,
        candidates,
        merge_distance_seconds=merge_distance_seconds,
    )
    if not clusters:
        return [], []

    pause_limit_seconds = min(1.25, max(0.75, float(merge_distance_seconds) * 0.2))
    assembled_clips: List[List[float]] = []
    cluster_details: List[Dict[str, Any]] = []

    for cluster_index, cluster in enumerate(clusters, start=1):
        start_index, end_index = _expand_cluster_to_transcript_boundaries(
            normalized_transcript,
            cluster,
            min_duration_seconds=min_duration_seconds,
            pause_limit_seconds=pause_limit_seconds,
        )
        start_segment = normalized_transcript[start_index]
        end_segment = normalized_transcript[end_index]
        if _segment_text_is_fragment(start_segment) or _segment_text_is_fragment(end_segment):
            continue
        if _starts_with_continuation(str(start_segment["text"])) or not _ends_with_sentence(str(end_segment["text"])):
            continue

        start = float(start_segment["boundary_start"])
        end = float(end_segment["boundary_end"])
        if end <= start:
            continue

        score = max(float(cluster["score"]), max(float(item.get("score", 0.0)) for item in cluster["sources"]))
        assembled_clips.append([start, end, score])
        cluster_details.append(
            {
                "cluster_index": cluster_index,
                "start_index": start_index,
                "end_index": end_index,
                "start": start,
                "end": end,
                "score": score,
                "source_count": len(cluster["sources"]),
                "sources": [dict(item) for item in cluster["sources"]],
            }
        )

    deduped_clips: List[List[float]] = []
    deduped_details: List[Dict[str, Any]] = []
    seen: set[Tuple[float, float]] = set()
    for clip, details in sorted(
        zip(assembled_clips, cluster_details),
        key=lambda item: (item[0][0], item[0][1], -item[0][2]),
    ):
        key = (round(float(clip[0]), 3), round(float(clip[1]), 3))
        if key in seen:
            continue
        seen.add(key)
        deduped_clips.append(clip)
        deduped_details.append(details)

    return deduped_clips, deduped_details


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


def _duration_tolerance_seconds(expected_duration: float) -> float:
    return max(0.35, min(1.0, max(0.0, float(expected_duration)) * 0.05))


def _probe_previous_keyframe_time(
    path: Path,
    ffprobe_bin: str,
    timestamp: float,
    *,
    search_window_seconds: float = 20.0,
) -> Optional[float]:
    window_start = max(0.0, float(timestamp) - max(1.0, float(search_window_seconds)))
    interval_length = max(1.0, float(timestamp) - window_start + 1.0)
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-skip_frame",
        "nokey",
        "-show_frames",
        "-show_entries",
        "frame=pts_time",
        "-print_format",
        "json",
        "-read_intervals",
        f"{_format_ffmpeg_time(window_start)}%+{_format_ffmpeg_time(interval_length)}",
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
        return None

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return None

    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        return None

    latest_keyframe: Optional[float] = None
    target_time = float(timestamp)
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        try:
            pts_time = float(frame.get("pts_time"))
        except (TypeError, ValueError):
            continue
        if pts_time > target_time + 1e-6:
            continue
        if latest_keyframe is None or pts_time > latest_keyframe:
            latest_keyframe = pts_time
    return latest_keyframe


def _copy_trim_start_is_accurate(
    requested_start: float,
    previous_keyframe_time: Optional[float],
    *,
    tolerance_seconds: float = 0.25,
) -> tuple[bool, str]:
    start_time = max(0.0, float(requested_start))
    tolerance = max(0.05, float(tolerance_seconds))
    if start_time <= tolerance:
        return True, ""
    if previous_keyframe_time is None:
        return False, "could not verify a nearby source keyframe"

    drift = max(0.0, start_time - float(previous_keyframe_time))
    if drift <= tolerance:
        return True, ""
    return False, f"nearest source keyframe is {drift:.3f}s before requested start"


def _verify_copy_trim_accuracy(
    source_video: Path,
    ffprobe_bin: str,
    *,
    requested_start: float,
    requested_duration: float,
) -> tuple[bool, str]:
    keyframe_time = _probe_previous_keyframe_time(
        source_video,
        ffprobe_bin,
        max(0.0, float(requested_start)),
    )
    return _copy_trim_start_is_accurate(
        requested_start,
        keyframe_time,
        tolerance_seconds=min(0.25, _duration_tolerance_seconds(requested_duration)),
    )


def _validate_media_output(
    path: Path,
    ffprobe_bin: str,
    *,
    expected_duration: Optional[float] = None,
) -> tuple[bool, str, Optional[float]]:
    if not path.exists():
        return False, "file was not created", None
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"unable to stat file: {exc}", None
    if size <= 0:
        return False, "file is empty", None

    duration = _probe_duration_seconds(path, ffprobe_bin, strict=False)
    if duration is None or duration <= 0:
        return False, "ffprobe reported invalid duration", duration
    if expected_duration is not None:
        tolerance = _duration_tolerance_seconds(expected_duration)
        if abs(duration - float(expected_duration)) > tolerance:
            return (
                False,
                f"duration mismatch (expected ~{float(expected_duration):.3f}s, got {duration:.3f}s)",
                duration,
            )
    return True, "", duration


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


def _accurate_trim_command(
    ffmpeg_bin: str,
    source_video: Path,
    output_path: Path,
    start: float,
    duration: float,
) -> List[str]:
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
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _find_existing_output(
    candidates: Sequence[Path],
    ffprobe_bin: str,
    logger: logging.Logger,
    *,
    expected_duration: Optional[float] = None,
) -> Optional[Path]:
    for candidate in candidates:
        if not candidate.exists():
            continue
        valid, reason, _ = _validate_media_output(
            candidate,
            ffprobe_bin,
            expected_duration=expected_duration,
        )
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
        valid, reason, _ = _validate_media_output(
            output_path,
            ffprobe_bin,
            expected_duration=duration,
        )
        if valid:
            accurate, accuracy_reason = _verify_copy_trim_accuracy(
                source_video,
                ffprobe_bin,
                requested_start=start,
                requested_duration=duration,
            )
            if not accurate:
                _delete_partial_output(output_path)
                errors.append(f"{output_path.suffix.lower()} copy trim failed: {accuracy_reason}")
                logger.warning(
                    "Stream-copy trim for %s was rejected because cut accuracy could not be guaranteed. Details: %s",
                    source_video.name,
                    accuracy_reason,
                )
                if attempt_index == 1:
                    logger.warning(
                        "MP4 stream-copy trim rejected for %s; retrying Matroska container before re-encode.",
                        source_video.name,
                    )
                continue
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

    reencode_output = mp4_output
    _delete_partial_output(reencode_output)
    logger.warning(
        "Falling back to accurate re-encode trim for %s because stream-copy could not guarantee a precise cut.",
        source_video.name,
    )
    succeeded, details = _run_ffmpeg_command(
        _accurate_trim_command(ffmpeg_bin, source_video, reencode_output, start, duration)
    )
    valid, reason, _ = _validate_media_output(
        reencode_output,
        ffprobe_bin,
        expected_duration=duration,
    )
    if succeeded and valid:
        logger.info("Accurate re-encode trim succeeded for %s (%s)", source_video.name, reencode_output.name)
        return reencode_output

    _delete_partial_output(reencode_output)
    errors.append(f"accurate re-encode trim failed: {reason or details}")
    raise RuntimeError("; ".join(errors))


def extract_clips(
    clips: Sequence[Sequence[float]],
    source_video: Path,
    output_dir: Path,
    logger: logging.Logger,
    progress_interval: int = 5,
    clip_name_indices: Optional[Sequence[int]] = None,
    skip_existing: bool = False,
    on_clip_done: Optional[Callable[[int, str, bool], None]] = None,
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
                (mp4_output, mkv_output),
                ffprobe_bin,
                logger,
                expected_duration=end - start,
            )
            if existing_output is not None:
                if on_clip_done is not None:
                    on_clip_done(clip_idx, str(existing_output), False)
                continue

        created_output = _extract_clip_stream_copy(
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            source_video=source_video,
            mp4_output=mp4_output,
            mkv_output=mkv_output,
            start=start,
            duration=end - start,
            logger=logger,
        )
        clip_count += 1
        if on_clip_done is not None:
            on_clip_done(clip_idx, str(created_output), True)
        if clip_count % max(1, progress_interval) == 0:
            logger.info("Extracted %s clips from %s", clip_count, source_video.name)

    return clip_count
