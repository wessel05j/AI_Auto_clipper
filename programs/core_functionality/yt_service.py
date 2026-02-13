from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import parse_qs, urlparse

import requests
import yt_dlp


PathLike = Union[str, Path]


class _SilentYDLLogger:
    def debug(self, msg: str) -> None:
        return None

    def warning(self, msg: str) -> None:
        return None

    def error(self, msg: str) -> None:
        return None


class YTService:
    """
    Standalone YouTube fetch + download helper.
    Uses browser/manual cookie detection with robust yt-dlp fallback strategies.
    """

    DEFAULT_BROWSERS = ("firefox", "edge", "chrome", "brave", "opera", "vivaldi")
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
    _VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")

    def __init__(
        self,
        base_dir: Optional[PathLike] = None,
        browser_priority: Optional[Sequence[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[2]
        self.browser_priority = tuple(browser_priority or self.DEFAULT_BROWSERS)
        self.logger = logger or logging.getLogger(__name__)
        self._cookies_checked = False
        self._cookies_file: Optional[Path] = None
        self._browser_cookies: Optional[Tuple[str, ...]] = None
        self._download_history_cache: Optional[set[str]] = None
        self._ensure_history_file()

    @property
    def download_history_file(self) -> Path:
        return self.base_dir / "system" / "downloaded_youtube_links.txt"

    def _ensure_history_file(self) -> None:
        history_file = self.download_history_file
        history_file.parent.mkdir(parents=True, exist_ok=True)
        if not history_file.exists():
            history_file.touch()

    @classmethod
    def _normalize_video_url(cls, raw_url: str) -> str:
        value = (raw_url or "").strip()
        if not value:
            return value

        # Raw video id form.
        if cls._VIDEO_ID_PATTERN.fullmatch(value):
            return f"https://www.youtube.com/watch?v={value}"

        try:
            parsed = urlparse(value)
        except Exception:
            return value

        netloc = parsed.netloc.lower()
        path = parsed.path or ""
        video_id: Optional[str] = None

        if "youtu.be" in netloc:
            candidate = path.strip("/").split("/")[0]
            if cls._VIDEO_ID_PATTERN.fullmatch(candidate):
                video_id = candidate
        elif "youtube.com" in netloc:
            if path == "/watch":
                query = parse_qs(parsed.query)
                candidate = (query.get("v") or [None])[0]
                if candidate and cls._VIDEO_ID_PATTERN.fullmatch(candidate):
                    video_id = candidate
            else:
                segments = [seg for seg in path.split("/") if seg]
                if len(segments) >= 2 and segments[0] in {"shorts", "live", "embed"}:
                    candidate = segments[1]
                    if cls._VIDEO_ID_PATTERN.fullmatch(candidate):
                        video_id = candidate

        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
        return value

    def _load_download_history(self) -> set[str]:
        if self._download_history_cache is not None:
            return set(self._download_history_cache)

        self._ensure_history_file()
        history: set[str] = set()
        with open(self.download_history_file, "r", encoding="utf-8") as file:
            for line in file:
                normalized = self._normalize_video_url(line.strip())
                if normalized:
                    history.add(normalized)

        self._download_history_cache = set(history)
        return history

    def _add_to_download_history(self, url: str) -> None:
        normalized = self._normalize_video_url(url)
        if not normalized:
            return

        history = self._load_download_history()
        if normalized in history:
            return

        self._ensure_history_file()
        with open(self.download_history_file, "a", encoding="utf-8") as file:
            file.write(f"{normalized}\n")

        history.add(normalized)
        self._download_history_cache = history

    def _is_in_download_history(self, url: str) -> bool:
        normalized = self._normalize_video_url(url)
        if not normalized:
            return False
        return normalized in self._load_download_history()

    def _find_cookie_file(self) -> Optional[Path]:
        candidates = [
            self.base_dir / "resources" / "cookies.txt",
            self.base_dir / "system" / "cookies.txt",
            self.base_dir / "cookies.txt",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        return None

    def _browser_cookie_source_is_valid(self, browser: str) -> bool:
        browser_tuple = (browser,)
        test_opts: Dict[str, Any] = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": "in_playlist",
            "playlist_items": "1",
            "cookiesfrombrowser": browser_tuple,
            "ignoreerrors": True,
            "no_warnings": True,
            "ignoreconfig": True,
            "logger": _SilentYDLLogger(),
        }
        try:
            with yt_dlp.YoutubeDL(test_opts) as ydl:
                result = ydl.extract_info(
                    "https://www.youtube.com/@YouTube/videos",
                    download=False,
                )
            if result and result.get("entries"):
                self.logger.info("Successfully using cookies from: %s", browser)
                return True
        except Exception as exc:
            error_msg = str(exc).lower()
            if "could not find" in error_msg:
                self.logger.debug("Browser %s not installed", browser)
                return False
            if "dpapi" in error_msg or "decrypt" in error_msg:
                self.logger.warning(
                    "Browser %s cookies encrypted; close that browser and retry.",
                    browser,
                )
                return False
            if "failed to load cookies" in error_msg:
                self.logger.debug("Browser %s cookie loading failed", browser)
                return False

            # Unknown failure mode: match the reference behavior and still try this browser.
            self.logger.info("Using browser %s with warning: %s", browser, exc)
            return True

        return False

    def _detect_cookies(self) -> None:
        if self._cookies_checked:
            return

        self._cookies_checked = True
        self._cookies_file = self._find_cookie_file()
        if self._cookies_file:
            self.logger.info("Using cookies file: %s", self._cookies_file)
            return

        for browser in self.browser_priority:
            if self._browser_cookie_source_is_valid(browser):
                self._browser_cookies = (browser,)
                return

        self.logger.warning("=" * 60)
        self.logger.warning("No browser cookies available.")
        self.logger.warning("Close all browsers and retry, or export cookies.txt manually.")
        self.logger.warning("=" * 60)

    def _apply_cookies_to_opts(self, ydl_opts: Dict[str, Any]) -> None:
        self._detect_cookies()
        ydl_opts.pop("cookiefile", None)
        ydl_opts.pop("cookiesfrombrowser", None)
        if self._cookies_file:
            ydl_opts["cookiefile"] = str(self._cookies_file)
        elif self._browser_cookies:
            ydl_opts["cookiesfrombrowser"] = self._browser_cookies

    @staticmethod
    def _resolve_channel_id(result: Dict[str, Any], channel_url: str) -> Optional[str]:
        candidates: List[str] = []
        for key in ("channel_id", "uploader_id", "id", "webpage_url", "channel_url"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        candidates.append(channel_url)

        for candidate in candidates:
            if candidate.startswith("UC") and len(candidate) >= 20:
                return candidate
            if candidate.startswith("UU") and len(candidate) >= 20:
                return f"UC{candidate[2:]}"
            match = re.search(r"/channel/(UC[\w-]+)", candidate)
            if match:
                return match.group(1)
        return None

    def fetch_recent_links(
        self,
        channels: Sequence[str],
        hours_limit: int = 24,
        playlistend: int = 15,
    ) -> List[str]:
        threshold = datetime.now() - timedelta(hours=max(0, hours_limit))
        links_found: List[str] = []
        seen: set[str] = set()
        downloaded_history = self._load_download_history()

        if downloaded_history:
            self.logger.info(
                "Loaded %s previously downloaded links from history.",
                len(downloaded_history),
            )

        list_opts: Dict[str, Any] = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": "in_playlist",
            "playlistend": 1,
            "ignoreerrors": True,
            "no_warnings": True,
            "socket_timeout": 30,
            "ignoreconfig": True,
            "logger": _SilentYDLLogger(),
        }
        self._apply_cookies_to_opts(list_opts)

        with yt_dlp.YoutubeDL(list_opts) as ydl:
            for channel_url in channels:
                self.logger.info("Checking channel: %s", channel_url)
                try:
                    result = ydl.extract_info(channel_url, download=False)
                except Exception as exc:
                    self.logger.warning("Failed to read channel %s (%s)", channel_url, exc)
                    continue

                if not result:
                    continue

                channel_id = self._resolve_channel_id(result, channel_url)
                if not channel_id:
                    self.logger.warning("Could not resolve channel ID for %s", channel_url)
                    continue

                rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
                try:
                    rss_response = requests.get(
                        rss_url,
                        timeout=15,
                        headers={"User-Agent": self.USER_AGENT},
                    )
                except Exception as exc:
                    self.logger.warning("RSS request failed for %s (%s)", channel_url, exc)
                    continue

                if rss_response.status_code != 200:
                    self.logger.warning(
                        "RSS request returned %s for %s",
                        rss_response.status_code,
                        channel_url,
                    )
                    continue

                videos_from_channel = 0
                entries = re.findall(r"<entry>.*?</entry>", rss_response.text, flags=re.DOTALL)
                for entry in entries:
                    video_id_match = re.search(r"<yt:videoId>([^<]+)</yt:videoId>", entry)
                    published_match = re.search(r"<published>([^<]+)</published>", entry)
                    if not video_id_match or not published_match:
                        continue

                    try:
                        publish_date = datetime.fromisoformat(
                            published_match.group(1).replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                    except ValueError:
                        continue

                    if publish_date < threshold:
                        break

                    video_url = self._normalize_video_url(
                        f"https://www.youtube.com/watch?v={video_id_match.group(1).strip()}"
                    )
                    if video_url in downloaded_history:
                        continue
                    if video_url in seen:
                        continue

                    seen.add(video_url)
                    links_found.append(video_url)
                    videos_from_channel += 1

                    if videos_from_channel >= max(1, playlistend):
                        break

                self.logger.info("  -> Found %s videos", videos_from_channel)

        return links_found

    @staticmethod
    def _new_files(before: List[Path], after: List[Path]) -> List[Path]:
        before_set = {str(path.resolve()) for path in before}
        new_files: List[Path] = []
        for path in after:
            if str(path.resolve()) in before_set:
                continue
            if path.suffix.lower() in {".part", ".ytdl"}:
                continue
            if path.suffix.lower() not in {".mp4", ".mkv", ".webm", ".mov"}:
                continue
            if path.is_file():
                new_files.append(path)
        return new_files

    def download_video(
        self,
        url: str,
        output_dir: PathLike,
        filename_template: str = "%(title).200B.%(ext)s",
    ) -> Optional[str]:
        normalized_url = self._normalize_video_url(url)
        if self._is_in_download_history(normalized_url):
            self.logger.info("Skipping already downloaded URL from history: %s", normalized_url)
            return None

        target_url = normalized_url or url
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        before_files = list(output_path.glob("*"))

        base_opts: Dict[str, Any] = {
            "merge_output_format": "mp4",
            "outtmpl": str(output_path / filename_template),
            "quiet": False,
            "noprogress": False,
            # Keep both runtimes enabled; whichever is available will be used.
            "js_runtimes": {"node": {}, "deno": {}},
            # Required for current yt-dlp n-challenge handling when yt_dlp_ejs is not installed.
            "remote_components": ["ejs:github"],
            "http_headers": {
                "User-Agent": self.USER_AGENT,
                "Accept-Language": "en-us,en;q=0.5",
            },
            "socket_timeout": 60,
            "retries": 20,
            "fragment_retries": 20,
            "skip_unavailable_fragments": True,
            "ignoreerrors": False,
            "allow_unplayable_formats": False,
            "ignoreconfig": True,
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
        }

        download_strategies: List[Dict[str, Any]] = [
            {
                "name": "cookies-default-clients",
                "use_cookies": True,
                "format": (
                    "best[height<=720][ext=mp4]/"
                    "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/"
                    "bestvideo[height<=720]+bestaudio/"
                    "best[height<=720]/best"
                ),
                "extractor_args": {
                    "youtube": {
                        "player_client": ["tv_downgraded", "web", "web_safari"],
                    }
                },
            },
            {
                "name": "cookies-mobile-clients",
                "use_cookies": True,
                "format": "bestvideo*+bestaudio/best",
                "extractor_args": {
                    "youtube": {
                        "player_client": ["ios_downgraded", "android_vr", "web"],
                    }
                },
            },
            {
                "name": "no-cookies-mobile-clients",
                "use_cookies": False,
                "format": "bestvideo*+bestaudio/best",
                "extractor_args": {
                    "youtube": {
                        "player_client": ["ios_downgraded", "android_vr"],
                    }
                },
            },
            {
                "name": "no-cookies-generic-fallback",
                "use_cookies": False,
                "format": "bestvideo*+bestaudio/best",
            },
        ]

        for strategy in download_strategies:
            ydl_opts = dict(base_opts)
            ydl_opts["format"] = strategy["format"]

            extractor_args = strategy.get("extractor_args")
            if extractor_args:
                ydl_opts["extractor_args"] = extractor_args
            else:
                ydl_opts.pop("extractor_args", None)

            if strategy.get("use_cookies", True):
                self._apply_cookies_to_opts(ydl_opts)
            else:
                ydl_opts.pop("cookiefile", None)
                ydl_opts.pop("cookiesfrombrowser", None)

            self.logger.info("Downloading with yt-dlp (%s): %s", strategy["name"], target_url)
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    result = ydl.download([target_url])
            except Exception as exc:
                self.logger.warning(
                    "Download strategy '%s' failed for %s (%s)",
                    strategy["name"],
                    target_url,
                    exc,
                )
                continue

            if result not in (None, 0):
                self.logger.warning(
                    "yt-dlp returned non-zero status for %s using strategy '%s': %s",
                    target_url,
                    strategy["name"],
                    result,
                )
                continue

            after_files = list(output_path.glob("*"))
            created_files = self._new_files(before_files, after_files)
            if not created_files:
                self.logger.warning(
                    "No video file created for %s using strategy '%s'",
                    target_url,
                    strategy["name"],
                )
                continue

            created_files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
            downloaded = created_files[0]
            if downloaded.stat().st_size < 10_000:
                self.logger.warning(
                    "Downloaded file too small (%s bytes), removing %s",
                    downloaded.stat().st_size,
                    downloaded.name,
                )
                downloaded.unlink(missing_ok=True)
                continue

            self._add_to_download_history(target_url)
            self.logger.info("Download strategy '%s' succeeded for %s", strategy["name"], target_url)
            return str(downloaded)

        return None
