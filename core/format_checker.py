from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FormatDecision:
    format_string: str
    strategy: str


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _is_video_only(fmt: Dict[str, Any]) -> bool:
    return fmt.get("vcodec") not in (None, "none") and fmt.get("acodec") in (None, "none")


def _is_progressive(fmt: Dict[str, Any]) -> bool:
    return fmt.get("vcodec") not in (None, "none") and fmt.get("acodec") not in (None, "none")


def _sort_key(fmt: Dict[str, Any]) -> tuple[int, float]:
    height = _to_int(fmt.get("height")) or 0
    bitrate = _to_float(fmt.get("tbr") or fmt.get("vbr"))
    return height, bitrate


def _pick_video_stream(formats: List[Dict[str, Any]], min_height: int, max_height: int) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for fmt in formats:
        if not _is_video_only(fmt):
            continue
        height = _to_int(fmt.get("height"))
        if height is None:
            continue
        if min_height <= height <= max_height:
            candidates.append(fmt)

    if not candidates:
        return None
    candidates.sort(key=_sort_key, reverse=True)
    return candidates[0]


def _pick_progressive(formats: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    progressive = [fmt for fmt in formats if _is_progressive(fmt)]
    if not progressive:
        return None
    progressive.sort(key=_sort_key, reverse=True)
    return progressive[0]


def choose_download_format(extracted_info: Dict[str, Any]) -> FormatDecision:
    """
    Select best format with this fallback chain:
    1) 2K-ish video (1081-1600p)
    2) best 1080p video
    3) best progressive (audio + video)
    4) generic yt-dlp bestvideo+bestaudio fallback
    """
    formats = extracted_info.get("formats") or []
    if not isinstance(formats, list) or not formats:
        return FormatDecision("bestvideo+bestaudio/best", "generic-fallback")

    two_k = _pick_video_stream(formats, min_height=1081, max_height=1600)
    if two_k and two_k.get("format_id"):
        fmt_id = str(two_k["format_id"])
        return FormatDecision(
            format_string=f"{fmt_id}+bestaudio[acodec!=none]/{fmt_id}+bestaudio/{fmt_id}/best",
            strategy="2k-video-plus-audio",
        )

    full_hd = _pick_video_stream(formats, min_height=1080, max_height=1080)
    if full_hd and full_hd.get("format_id"):
        fmt_id = str(full_hd["format_id"])
        return FormatDecision(
            format_string=f"{fmt_id}+bestaudio[acodec!=none]/{fmt_id}+bestaudio/{fmt_id}/best",
            strategy="1080p-video-plus-audio",
        )

    progressive = _pick_progressive(formats)
    if progressive and progressive.get("format_id"):
        return FormatDecision(str(progressive["format_id"]), "progressive-best")

    return FormatDecision("bestvideo+bestaudio/best", "generic-fallback")
