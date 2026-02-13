import os
from typing import Optional

from programs.core_functionality.yt_service import YTService


_YOUTUBE_CLIENT: Optional[YTService] = None


def _youtube_client() -> YTService:
    global _YOUTUBE_CLIENT
    if _YOUTUBE_CLIENT is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        _YOUTUBE_CLIENT = YTService(base_dir=base_dir)
    return _YOUTUBE_CLIENT


def yt_downloader(url: str, output: str) -> Optional[str]:
    os.makedirs(output, exist_ok=True)
    downloaded = _youtube_client().download_video(url=url, output_dir=output)
    return downloaded
