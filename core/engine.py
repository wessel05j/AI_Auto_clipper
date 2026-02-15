from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
import gc
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
import requests

from core.ai_pipeline import AIPipeline, chunk_transcript
from core.clipping import (
    extract_clips,
    merge_segments,
    scan_input_videos,
    transcribe_video,
)
from core.yt_handler import YTHandler
from ui.components import hard_clear, show_header
from utils.model_selector import ensure_ollama_running, model_billions
from utils.validators import config_paths, load_optional_profile, save_json_file


class _OptionalSpinnerColumn(SpinnerColumn):
    def render(self, task) -> Text:
        if task.fields.get("plain"):
            return Text("")
        return super().render(task)


class _OptionalBarColumn(BarColumn):
    def render(self, task) -> Text:
        if task.fields.get("plain"):
            return Text("")
        return super().render(task)


class _OptionalTaskProgressColumn(TaskProgressColumn):
    def render(self, task) -> Text:
        if task.fields.get("plain"):
            return Text("")
        return super().render(task)


class _OptionalTimeElapsedColumn(TimeElapsedColumn):
    def render(self, task) -> Text:
        if task.fields.get("plain"):
            return Text("")
        return super().render(task)


class _OptionalTimeRemainingColumn(TimeRemainingColumn):
    def render(self, task) -> Text:
        if task.fields.get("plain"):
            return Text("")
        return super().render(task)


class ClippingEngine:
    """Main runtime engine that orchestrates download, transcription, AI scan, and clip extraction."""

    def __init__(self, base_dir: Path, config: Dict[str, Any], logger: logging.Logger, console: Console) -> None:
        self.base_dir = base_dir
        self.config = config
        self.logger = logger
        self.console = console
        self.yt_handler = YTHandler(base_dir=base_dir, logger=logger)

        paths_cfg = config.get("paths", {})
        self.input_dir = base_dir / str(paths_cfg.get("input_dir", "input"))
        self.output_dir = base_dir / str(paths_cfg.get("output_dir", "output"))
        self.temp_dir = base_dir / str(paths_cfg.get("temp_dir", "temp"))
        self.system_dir = base_dir / str(paths_cfg.get("system_dir", "system"))
        self.status_file = self.system_dir / "status.json"
        self._config_paths = config_paths(self.base_dir)
        self.hardware_profile = load_optional_profile(self._config_paths["hardware_file"])

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.system_dir.mkdir(parents=True, exist_ok=True)

    def _persist_config(self) -> None:
        save_json_file(self._config_paths["config_file"], self.config)

    def _write_status(self, payload: Dict[str, Any]) -> None:
        save_json_file(self.status_file, payload)

    def _load_whisper_model(self):
        import whisper
        import torch

        transcription_model = str(self.config["transcription"]["model"]).lower()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info("Loading Whisper model '%s' on %s", transcription_model, device)
        return whisper.load_model(transcription_model, device=device)

    def _ensure_ollama(self) -> bool:
        ollama_url = str(self.config["ollama"]["url"])
        ok = ensure_ollama_running(ollama_url=ollama_url, logger=self.logger)
        if not ok:
            self.logger.error("Ollama did not become ready at %s", ollama_url)
            return False
        return True

    def _download_pending_links(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        links = list(self.config["clipping"].get("youtube_links", []))
        total_links = len(links)
        if progress_callback:
            progress_callback(0, total_links, "starting")
        if not links:
            return

        self.logger.info("Downloading %s queued YouTube links", len(links))
        remaining: List[str] = []
        for index, link in enumerate(links, start=1):
            normalized_link = self.yt_handler.normalize_video_url(str(link))
            if not normalized_link:
                self.logger.warning("Skipping empty/invalid YouTube link: %s", link)
                if progress_callback:
                    progress_callback(index, total_links, "invalid link")
                continue
            try:
                if progress_callback:
                    progress_callback(index - 1, total_links, f"downloading {index}/{total_links}")
                downloaded = self.yt_handler.download_video(url=normalized_link, output_dir=self.input_dir)
            except Exception as exc:
                self.logger.warning("Error downloading %s: %s", normalized_link, exc)
                remaining.append(normalized_link)
                if progress_callback:
                    progress_callback(index, total_links, "failed")
                continue

            if downloaded:
                self.logger.info("Downloaded: %s", downloaded)
                if progress_callback:
                    progress_callback(index, total_links, "downloaded")
            else:
                self.logger.info("Skipped or failed: %s", normalized_link)
                if progress_callback:
                    progress_callback(index, total_links, "skipped")

        self.config["clipping"]["youtube_links"] = remaining
        self._persist_config()
        if progress_callback:
            progress_callback(total_links, total_links, "completed")

    def _build_ai_pipeline(self) -> AIPipeline:
        return AIPipeline(
            model=str(self.config["ollama"]["model"]),
            ollama_url=str(self.config["ollama"]["url"]),
            system_prompt=str(self.config["clipping"]["system_prompt"]),
            temperature=float(self.config["ollama"]["temperature"]),
            max_output_tokens=int(self.config["ollama"]["max_output_tokens"]),
            max_context_tokens=int(self.config["runtime"]["total_context_tokens"]),
            logger=self.logger,
        )

    def _archive_video(self, source_video: Path) -> None:
        target = self.temp_dir / source_video.name
        if target.exists():
            stem = source_video.stem
            suffix = source_video.suffix
            counter = 1
            while True:
                candidate = self.temp_dir / f"{stem}_{counter:03d}{suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                counter += 1
        shutil.move(str(source_video), str(target))

    def _temp_files(self) -> List[Path]:
        return sorted(
            [path for path in self.temp_dir.rglob("*") if path.is_file()],
            key=lambda path: path.stat().st_mtime,
        )

    def _cleanup_temp_by_size(self, max_size_gb: float) -> int:
        max_bytes = int(max(0.1, float(max_size_gb)) * (1024**3))
        files = self._temp_files()
        total_bytes = sum(path.stat().st_size for path in files)
        if total_bytes <= max_bytes:
            return 0

        deleted = 0
        for file_path in files:
            if total_bytes <= max_bytes:
                break
            try:
                file_size = file_path.stat().st_size
                file_path.unlink(missing_ok=True)
                total_bytes -= file_size
                deleted += 1
            except Exception as exc:
                self.logger.debug("Temp cleanup could not remove %s: %s", file_path, exc)
        return deleted

    def _cleanup_temp_by_age(self, max_age_days: int) -> int:
        cutoff = datetime.now() - timedelta(days=max(1, int(max_age_days)))
        deleted = 0
        for file_path in self._temp_files():
            try:
                modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            except Exception:
                continue
            if modified >= cutoff:
                continue
            try:
                file_path.unlink(missing_ok=True)
                deleted += 1
            except Exception as exc:
                self.logger.debug("Temp cleanup could not remove %s: %s", file_path, exc)
        return deleted

    def _apply_temp_cleanup_policy(self) -> None:
        cleanup = self.config.get("maintenance", {}).get("temp_cleanup", {})
        if not isinstance(cleanup, dict):
            return

        mode = str(cleanup.get("mode", "never")).strip().lower()
        if mode == "never":
            return

        deleted = 0
        if mode == "max_size_gb":
            deleted = self._cleanup_temp_by_size(float(cleanup.get("max_size_gb", 20)))
        elif mode == "max_age_days":
            deleted = self._cleanup_temp_by_age(int(cleanup.get("max_age_days", 30)))

        if deleted > 0:
            self.logger.info("Temp cleanup removed %s file(s) using mode '%s'.", deleted, mode)

    @contextmanager
    def _mute_console_logs(self) -> Iterator[None]:
        root_logger = logging.getLogger()
        handler_levels: List[tuple[logging.Handler, int]] = []
        for handler in root_logger.handlers:
            stream = getattr(handler, "stream", None)
            if stream in (sys.stdout, sys.stderr):
                handler_levels.append((handler, handler.level))
                handler.setLevel(logging.CRITICAL + 1)
        try:
            yield
        finally:
            for handler, level in handler_levels:
                handler.setLevel(level)

    @staticmethod
    def _extract_min_duration_seconds(user_query: str) -> float:
        text = str(user_query or "").lower()
        patterns = (
            (r"(?:at\s*least|atleast|minimum|min(?:imum)?|no\s+less\s+than)\s*(\d+(?:\.\d+)?)\s*(seconds?|secs?|sec|s)\b", 1.0),
            (r"(?:at\s*least|atleast|minimum|min(?:imum)?|no\s+less\s+than)\s*(\d+(?:\.\d+)?)\s*(minutes?|mins?|min|m)\b", 60.0),
            (r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|sec|s)\s*(?:minimum|min|or\s+more|\+)", 1.0),
            (r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|min|m)\s*(?:minimum|min|or\s+more|\+)", 60.0),
        )
        for pattern, scale in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            try:
                value = float(match.group(1))
            except Exception:
                continue
            if value > 0:
                return value * scale
        return 0.0

    @classmethod
    def _build_effective_query(cls, user_query: str) -> tuple[str, float]:
        min_duration = cls._extract_min_duration_seconds(user_query)

        constraints: List[str] = []
        if min_duration > 0:
            constraints.append(f"- Minimum clip duration: {min_duration:.1f} seconds.")
            constraints.append("- 'At least' is a lower bound, not an exact target duration.")
            constraints.append("- Prefer longer continuous relevant sections when they remain on-topic.")
            constraints.append("- Avoid cutting mid-sentence; keep natural spoken boundaries.")

        if not constraints:
            return str(user_query), min_duration

        constrained_query = (
            f"{str(user_query).strip()}\n\n"
            "Strict constraints:\n"
            f"{chr(10).join(constraints)}"
        )
        return constrained_query, min_duration

    @staticmethod
    def _filter_min_duration(
        clips: List[List[float]],
        min_duration_seconds: float,
    ) -> tuple[List[List[float]], int]:
        if min_duration_seconds <= 0:
            return clips, 0
        filtered: List[List[float]] = []
        rejected = 0
        for clip in clips:
            if len(clip) < 3:
                continue
            start = float(clip[0])
            end = float(clip[1])
            score = float(clip[2])
            if (end - start) < min_duration_seconds:
                rejected += 1
                continue
            filtered.append([start, end, score])
        return filtered, rejected

    @staticmethod
    def _estimate_transcription_progress(elapsed_seconds: float) -> tuple[int, str]:
        # Estimated progress to keep UI alive while whisper transcribes in one blocking call.
        if elapsed_seconds < 1.6:
            stage = "loading audio"
        elif elapsed_seconds < 5.0:
            stage = "running whisper"
        elif elapsed_seconds < 12.0:
            stage = "decoding speech"
        else:
            stage = "finalizing transcript"

        progress_percent = int(min(96.0, 8.0 + (elapsed_seconds * 3.1)))
        return max(1, progress_percent), stage

    def _render_engine_frame(self, subtitle: Optional[str] = None) -> None:
        hard_clear(self.console)
        show_header(self.console)
        self.console.rule("[bold cyan]Clipping Engine")
        if subtitle:
            self.console.print(f"[cyan]{subtitle}[/cyan]")

    def _resolve_runtime_chunk_cap(
        self,
        model_name: str,
        configured_chunk_tokens: int,
    ) -> tuple[int, float]:
        """
        Apply runtime token safety based on model size and available hardware budget.
        Returns: (effective_chunk_tokens, pressure_ratio)
        pressure_ratio > 1 means model estimate exceeds budget.
        """
        effective = int(max(900, configured_chunk_tokens))
        billions = model_billions(model_name)

        if billions is not None:
            if billions <= 8.0:
                effective = min(effective, 1800)
            elif billions <= 14.0:
                effective = min(effective, 2400)
            elif billions <= 20.0:
                effective = min(effective, 3000)
            else:
                effective = min(effective, 3600)

        gpu_vram_gb = float(self.hardware_profile.get("gpu_vram_gb", 0.0) or 0.0)
        ram_gb = float(self.hardware_profile.get("ram_gb", 0.0) or 0.0)
        intensity = str(self.config.get("runtime", {}).get("setup_intensity", "balanced")).lower().strip()
        if gpu_vram_gb > 0:
            multiplier = {"light": 0.45, "balanced": 0.64, "maximum": 0.82}.get(intensity, 0.64)
            budget_gb = max(2.0, gpu_vram_gb * multiplier)
        else:
            multiplier = {"light": 0.18, "balanced": 0.28, "maximum": 0.40}.get(intensity, 0.28)
            budget_gb = max(2.0, ram_gb * multiplier)

        estimated_model_gb = max(1.5, (billions or 10.0) * 0.62)
        pressure_ratio = estimated_model_gb / max(1.0, budget_gb)
        if pressure_ratio >= 1.2:
            effective = min(effective, 1300)
        elif pressure_ratio >= 1.0:
            effective = min(effective, 1600)
        elif pressure_ratio >= 0.8:
            effective = min(effective, 2100)

        return max(900, int(effective)), float(pressure_ratio)

    @staticmethod
    def _run_with_activity_heartbeat(
        progress: Progress,
        activity_task_id: Any,
        base_description: str,
        action: Callable[[], Any],
        interval_seconds: float = 0.25,
        heartbeat_callback: Optional[Callable[[float, int], None]] = None,
    ) -> Any:
        started_at = time.monotonic()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(action)
            pulse = 0
            while not future.done():
                time.sleep(max(0.05, float(interval_seconds)))
                pulse = (pulse + 1) % 4
                elapsed = max(0.0, time.monotonic() - started_at)
                if heartbeat_callback is not None:
                    heartbeat_callback(elapsed, pulse)
                else:
                    dots = "." * pulse
                    progress.update(activity_task_id, description=f"{base_description}{dots}")
                    progress.refresh()
            return future.result()

    @staticmethod
    def _augment_chunks_with_bridges(
        chunks: List[List[List[Any]]],
        max_bridge_segments: int = 4,
    ) -> tuple[List[List[List[Any]]], int]:
        if len(chunks) <= 1:
            return chunks, 0

        edge_segments = max(1, int(max_bridge_segments))
        bridges: List[List[List[Any]]] = []
        for index in range(len(chunks) - 1):
            left = chunks[index][-edge_segments:]
            right = chunks[index + 1][:edge_segments]
            bridge = [list(item) for item in (left + right) if len(item) >= 3]
            if len(bridge) >= 2:
                bridges.append(bridge)

        combined = chunks + bridges
        deduped: List[List[List[Any]]] = []
        seen: set[tuple[float, float, int]] = set()
        for chunk in combined:
            if not chunk:
                continue
            try:
                key = (round(float(chunk[0][0]), 3), round(float(chunk[-1][1]), 3), len(chunk))
            except Exception:
                key = (0.0, 0.0, len(chunk))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)
        return deduped, len(bridges)

    def _offload_ollama_model(self) -> None:
        model_name = str(self.config.get("ollama", {}).get("model", "")).strip()
        ollama_url = str(self.config.get("ollama", {}).get("url", "http://localhost:11434")).rstrip("/")
        if not model_name:
            return

        try:
            ps_response = requests.get(f"{ollama_url}/api/ps", timeout=10)
            loaded_models = []
            if ps_response.ok:
                payload = ps_response.json()
                loaded_models = [str(item.get("model", "")).lower() for item in payload.get("models", [])]
            if model_name.lower() not in loaded_models:
                self.logger.info("Ollama model `%s` is not currently loaded; no offload needed.", model_name)
                return
        except Exception as exc:
            self.logger.debug("Could not query loaded Ollama models: %s", exc)

        try:
            subprocess.run(
                ["ollama", "stop", model_name],
                check=False,
                capture_output=True,
                timeout=20,
            )
            self.logger.info("Offloaded Ollama model `%s` via `ollama stop`.", model_name)
            return
        except Exception as exc:
            self.logger.debug("Fallback `ollama stop` failed: %s", exc)

        try:
            payload = {
                "model": model_name,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            }
            response = requests.post(
                f"{ollama_url}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=20,
            )
            if response.ok:
                self.logger.info("Offloaded Ollama model `%s` via keep_alive=0.", model_name)
        except Exception as exc:
            self.logger.debug("Ollama API keep_alive unload failed: %s", exc)

    def _offload_runtime_resources(self, whisper_model: Any) -> None:
        try:
            if whisper_model is not None:
                del whisper_model
            gc.collect()
        except Exception:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                self.logger.info("Cleared CUDA cache after engine run.")
        except Exception:
            pass

        self._offload_ollama_model()

    def run(self) -> None:
        self._render_engine_frame()
        self.logger.info("Clipping engine run initialized")

        status: Dict[str, Any] = {
            "total_videos": 0,
            "progress_percent": 0,
            "current_video": None,
            "current_step": "starting",
        }
        run_stats: Dict[str, int] = {
            "ai_clipping_base_chunks": 0,
            "ai_clipping_bridge_chunks": 0,
            "ai_clipping_chunks_scanned": 0,
            "ai_clipping_candidates": 0,
            "merged_candidates": 0,
            "clips_extracted": 0,
        }

        whisper_model: Any = None
        try:
            with self._mute_console_logs():
                with Progress(
                    _OptionalSpinnerColumn(style="cyan"),
                    TextColumn("[progress.description]{task.description}"),
                    _OptionalBarColumn(bar_width=42),
                    _OptionalTaskProgressColumn(),
                    _OptionalTimeElapsedColumn(),
                    _OptionalTimeRemainingColumn(),
                    console=self.console,
                    transient=True,
                    ) as progress:
                    queued_links = list(self.config["clipping"].get("youtube_links", []))
                    queued_count = len(queued_links)
                    links_task = progress.add_task(
                        f"[yellow]YouTube links: 0/{queued_count}",
                        total=max(1, queued_count),
                        completed=0,
                    )
                    videos_task = progress.add_task(
                        "[cyan]Videos: waiting",
                        total=1,
                        completed=0,
                    )
                    activity_task = progress.add_task(
                        "[green]Activity: Booting",
                        total=1,
                        completed=0,
                    )
                    ai_clipping_task = progress.add_task(
                        "[magenta]AI clipping chunks: waiting",
                        total=1,
                        completed=0,
                        start=False,
                        visible=False,
                    )
                    activity_panel_title_task = progress.add_task(
                        "[bright_green]────────────[/bright_green][bold bright_cyan] Activity Panel [/bold bright_cyan][bright_green]────────────[/bright_green]",
                        total=1,
                        completed=1,
                        plain=True,
                    )
                    panel_tasks = {
                        "ai_clipping_base_chunks": progress.add_task(
                            "[bright_green]Base chunks: 0[/bright_green]",
                            total=1,
                            completed=1,
                            plain=True,
                        ),
                        "ai_clipping_bridge_chunks": progress.add_task(
                            "[bright_green]Bridge chunks: 0[/bright_green]",
                            total=1,
                            completed=1,
                            plain=True,
                        ),
                        "ai_clipping_chunks_scanned": progress.add_task(
                            "[bright_green]Chunks scanned: 0[/bright_green]",
                            total=1,
                            completed=1,
                            plain=True,
                        ),
                        "ai_clipping_candidates": progress.add_task(
                            "[bright_green]Clips found: 0[/bright_green]",
                            total=1,
                            completed=1,
                            plain=True,
                        ),
                        "merged_candidates": progress.add_task(
                            "[bright_green]Candidates after merge: 0[/bright_green]",
                            total=1,
                            completed=1,
                            plain=True,
                        ),
                        "clips_extracted": progress.add_task(
                            "[bright_green]Clips extracted: 0[/bright_green]",
                            total=1,
                            completed=1,
                            plain=True,
                        ),
                        "output_folder": progress.add_task(
                            f"[bright_green]Output folder: {self.output_dir}[/bright_green]",
                            total=1,
                            completed=1,
                            plain=True,
                        ),
                    }

                    def refresh_activity_panel() -> None:
                        bridge_chunks_enabled = bool(self.config.get("runtime", {}).get("enable_bridge_chunks", True))
                        progress.update(
                            activity_panel_title_task,
                            description=(
                                "[bright_green]━━━━━━━━━━━━[/bright_green]"
                                "[bold bright_cyan] Activity Panel [/bold bright_cyan]"
                                "[bright_green]━━━━━━━━━━━━[/bright_green]"
                            ),
                        )
                        progress.update(
                            panel_tasks["ai_clipping_base_chunks"],
                            description=f"[bright_green]Base chunks: {run_stats['ai_clipping_base_chunks']}[/bright_green]",
                        )
                        progress.update(
                            panel_tasks["ai_clipping_bridge_chunks"],
                            description=f"[bright_green]Bridge chunks: {run_stats['ai_clipping_bridge_chunks']}[/bright_green]",
                            visible=bridge_chunks_enabled,
                        )
                        progress.update(
                            panel_tasks["ai_clipping_chunks_scanned"],
                            description=f"[bright_green]Chunks scanned: {run_stats['ai_clipping_chunks_scanned']}[/bright_green]",
                        )
                        progress.update(
                            panel_tasks["ai_clipping_candidates"],
                            description=f"[bright_green]Clips found: {run_stats['ai_clipping_candidates']}[/bright_green]",
                        )
                        progress.update(
                            panel_tasks["merged_candidates"],
                            description=f"[bright_green]Candidates after merge: {run_stats['merged_candidates']}[/bright_green]",
                        )
                        progress.update(
                            panel_tasks["clips_extracted"],
                            description=f"[bright_green]Clips extracted: {run_stats['clips_extracted']}[/bright_green]",
                        )
                        progress.update(
                            panel_tasks["output_folder"],
                            description=f"[bright_green]Output folder: {self.output_dir}[/bright_green]",
                        )
                        progress.refresh()

                    refresh_activity_panel()
                    self._write_status(status)
                    status["current_step"] = "checking_ollama"
                    self._write_status(status)
                    progress.update(activity_task, description="[green]Activity: Checking Ollama", total=1, completed=0)
                    progress.refresh()
                    if not self._ensure_ollama():
                        self.console.print("[bold red]Ollama is not available. Aborting engine run.[/bold red]")
                        return
                    progress.update(activity_task, completed=1)
    
                    def on_download_progress(completed: int, total: int, state: str) -> None:
                        display_total = max(1, total)
                        display_completed = min(max(0, completed), display_total)
                        if total <= 0:
                            links_label = "[yellow]YouTube links: 0/0 (none queued)"
                            activity_label = "[green]Activity: Downloading YouTube links (none queued)"
                        else:
                            links_label = f"[yellow]YouTube links: {min(completed, total)}/{total} ({state})"
                            activity_label = (
                                f"[green]Activity: Downloading YouTube links {min(completed, total)}/{total}"
                            )
                        progress.update(
                            links_task,
                            total=display_total,
                            completed=display_completed,
                            description=links_label,
                        )
                        progress.update(
                            activity_task,
                            total=display_total,
                            completed=display_completed,
                            description=activity_label,
                        )
                        progress.refresh()
    
                    status["current_step"] = "downloading_links"
                    self._write_status(status)
                    self._download_pending_links(progress_callback=on_download_progress)
                    if queued_count == 0:
                        progress.update(links_task, total=1, completed=1, description="[yellow]YouTube links: 0/0")
                        progress.update(
                            activity_task,
                            total=1,
                            completed=1,
                            description="[green]Activity: No YouTube downloads queued",
                        )
    
                    self._apply_temp_cleanup_policy()
    
                    status["current_step"] = "scanning_input"
                    self._write_status(status)
                    progress.update(activity_task, description="[green]Activity: Scanning input videos", total=1, completed=0)
                    progress.refresh()
                    videos = scan_input_videos(self.input_dir)
                    progress.update(activity_task, completed=1)
                    if not videos:
                        progress.update(videos_task, description="[cyan]Videos: 0/0", total=1, completed=1)
                        self.console.print("[yellow]No videos found in input/. Add files or fetch links first.[/yellow]")
                        return
    
                    status["total_videos"] = len(videos)
                    status["progress_percent"] = 0
                    self._write_status(status)
                    progress.update(videos_task, description=f"[cyan]Videos: 0/{len(videos)}", total=len(videos), completed=0)
    
                    status["current_step"] = "loading_whisper"
                    self._write_status(status)
                    progress.update(activity_task, description="[green]Activity: Loading Whisper model", total=1, completed=0)
                    progress.refresh()
                    try:
                        whisper_model = self._load_whisper_model()
                    except Exception as exc:
                        self.logger.error("Could not load Whisper model: %s", exc)
                        self.console.print(f"[red]Whisper load failed: {exc}[/red]")
                        return
                    progress.update(activity_task, completed=1)
    
                    ai_pipeline = self._build_ai_pipeline()
                    merge_distance = float(self.config["clipping"]["merge_distance_seconds"])
                    ai_loops = int(self.config["clipping"]["ai_loops"])
                    user_query = str(self.config["clipping"]["user_query"])
                    effective_query, min_duration_seconds = self._build_effective_query(user_query)
                    model_name = str(self.config["ollama"]["model"])
                    configured_chunk_tokens = int(self.config["runtime"]["max_chunk_tokens"])
                    chunk_overlap_segments = int(self.config.get("runtime", {}).get("chunk_overlap_segments", 3))
                    enable_bridge_chunks = bool(self.config.get("runtime", {}).get("enable_bridge_chunks", True))
                    bridge_chunk_edge_segments = int(self.config.get("runtime", {}).get("bridge_chunk_edge_segments", 4))
                    max_chunk_tokens, pressure_ratio = self._resolve_runtime_chunk_cap(
                        model_name=model_name,
                        configured_chunk_tokens=configured_chunk_tokens,
                    )
                    progress_interval = int(self.config["clipping"].get("clip_progress_interval", 5))
                    if min_duration_seconds > 0:
                        self.logger.info("Applying minimum clip duration: %.1f seconds", min_duration_seconds)
                    if max_chunk_tokens != configured_chunk_tokens:
                        self.logger.info(
                            "Adjusted chunk token cap from %s to %s for model/hardware pressure (ratio=%.2f).",
                            configured_chunk_tokens,
                            max_chunk_tokens,
                            pressure_ratio,
                        )
    
                    for index, video in enumerate(videos, start=1):
                        progress.update(
                            videos_task,
                            description=f"[cyan]Videos: {index - 1}/{len(videos)}",
                        )
                        progress.reset(activity_task, total=7, completed=0)
                        progress.update(
                            ai_clipping_task,
                            visible=False,
                            total=1,
                            completed=0,
                            description="[magenta]AI clipping chunks: waiting",
                        )
                        progress.update(
                            activity_task,
                            description=f"[green]Activity: Preparing {index}/{len(videos)} {video.name}",
                        )
    
                        status["current_video"] = video.name
                        status["progress_percent"] = round(((index - 1) / len(videos)) * 100, 1)
                        status["current_step"] = "transcribing"
                        self._write_status(status)
                        progress.update(activity_task, description="[green]Activity: Transcribing")
                        progress.refresh()

                        try:
                            transcript = transcribe_video(
                                video_path=video,
                                whisper_model=whisper_model,
                                logger=self.logger,
                            )
                            progress.advance(activity_task)
                            if not transcript:
                                self.logger.warning("Empty transcript for %s", video.name)
                                continue

                            status["current_step"] = "chunking"
                            self._write_status(status)
                            progress.update(activity_task, description="[green]Activity: Chunking transcript")
                            progress.refresh()
                            base_chunks = chunk_transcript(
                                transcript=transcript,
                                max_tokens=max_chunk_tokens,
                                overlap_segments=max(0, chunk_overlap_segments),
                            )
                            if enable_bridge_chunks:
                                chunks, bridge_count = self._augment_chunks_with_bridges(
                                    base_chunks,
                                    max_bridge_segments=max(1, bridge_chunk_edge_segments),
                                )
                            else:
                                chunks, bridge_count = base_chunks, 0
                            run_stats["ai_clipping_base_chunks"] += len(base_chunks)
                            run_stats["ai_clipping_bridge_chunks"] += int(bridge_count)
                            refresh_activity_panel()
                            progress.advance(activity_task)
                            if not chunks:
                                self.logger.warning("No chunks created for %s", video.name)
                                continue

                            status["current_step"] = "ai_scanning"
                            self._write_status(status)
                            progress.update(activity_task, description="[green]Activity: AI clipping scanning")
                            progress.refresh()
                            total_scan_steps = max(1, len(chunks) * max(1, ai_loops))
                            progress.reset(
                                ai_clipping_task,
                                total=total_scan_steps,
                                completed=0,
                                start=True,
                            )
                            progress.update(
                                ai_clipping_task,
                                visible=True,
                                description=f"[magenta]AI clipping chunks: 0/{total_scan_steps}",
                            )
                            scanned_baseline = int(run_stats["ai_clipping_chunks_scanned"])
                            if enable_bridge_chunks:
                                ai_clipping_scan_context = f"({len(base_chunks)} base + {bridge_count} bridge chunks)"
                            else:
                                ai_clipping_scan_context = f"({len(base_chunks)} base chunks)"
                            progress.update(
                                activity_task,
                                description=(
                                    "[green]Activity: AI clipping scanning "
                                    f"{ai_clipping_scan_context}"
                                ),
                            )
    
                            def on_ai_clipping_progress(
                                chunk_index: int,
                                total_chunks: int,
                                loop_index: int,
                                total_loops: int,
                                clips_found_in_step: int,
                            ) -> None:
                                completed_steps = ((chunk_index - 1) * total_loops) + loop_index
                                run_stats["ai_clipping_chunks_scanned"] = (
                                    scanned_baseline + min(completed_steps, total_scan_steps)
                                )
                                run_stats["ai_clipping_candidates"] += max(0, int(clips_found_in_step))
                                progress.update(
                                    activity_task,
                                    description=(
                                        "[green]Activity: AI clipping chunk "
                                        f"{chunk_index}/{total_chunks} (loop {loop_index}/{total_loops})"
                                    ),
                                )
                                progress.update(
                                    ai_clipping_task,
                                    completed=min(completed_steps, total_scan_steps),
                                    description=f"[magenta]AI clipping chunks: {min(completed_steps, total_scan_steps)}/{total_scan_steps}",
                                )
                                refresh_activity_panel()
    
                            raw_clip_groups = ai_pipeline.scan_all_chunks(
                                chunks=chunks,
                                user_query=effective_query,
                                ai_loops=ai_loops,
                                progress_callback=on_ai_clipping_progress,
                            )
                            refresh_activity_panel()
                            progress.update(ai_clipping_task, visible=False)
                            progress.advance(activity_task)

                            status["current_step"] = "merging_segments"
                            self._write_status(status)
                            progress.update(activity_task, description="[green]Activity: Merging segments")
                            progress.refresh()
                            merged_clips = merge_segments(raw_clip_groups, tolerance_seconds=merge_distance)
                            run_stats["merged_candidates"] += len(merged_clips)
                            refresh_activity_panel()
                            progress.advance(activity_task)

                            status["current_step"] = "filtering_candidates"
                            self._write_status(status)
                            progress.update(
                                activity_task,
                                description="[green]Activity: Filtering merged candidates",
                            )
                            progress.refresh()
                            final_clips, rejected_short = self._filter_min_duration(
                                merged_clips,
                                min_duration_seconds=min_duration_seconds,
                            )
                            if rejected_short > 0:
                                self.logger.debug(
                                    "Removed %s merged candidates shorter than %.1f seconds.",
                                    rejected_short,
                                    min_duration_seconds,
                                )
                            progress.advance(activity_task)

                            status["current_step"] = "extracting_clips"
                            self._write_status(status)
                            if not final_clips:
                                progress.advance(activity_task)
                                progress.update(activity_task, description="[green]Activity: Archiving source")
                                progress.refresh()
                                self._archive_video(video)
                                self._apply_temp_cleanup_policy()
                                progress.advance(activity_task)
                                continue
    
                            progress.update(activity_task, description="[green]Activity: Extracting clips")
                            progress.refresh()
                            created = extract_clips(
                                clips=final_clips,
                                source_video=video,
                                output_dir=self.output_dir,
                                logger=self.logger,
                                progress_interval=progress_interval,
                            )
                            run_stats["clips_extracted"] += int(created)
                            refresh_activity_panel()
                            self.logger.info(
                                "Extracted %s clips from %s (candidates=%s after_filter=%s)",
                                created,
                                video.name,
                                len(merged_clips),
                                len(final_clips),
                            )
                            progress.advance(activity_task)
                            progress.update(activity_task, description="[green]Activity: Archiving source")
                            progress.refresh()
                            self._archive_video(video)
                            self._apply_temp_cleanup_policy()
                            progress.advance(activity_task)
                        except Exception as exc:
                            self.logger.exception("Error processing %s: %s", video.name, exc)
                            self.console.print(f"[red]Failed processing {video.name}: {exc}[/red]")
                        finally:
                            status["progress_percent"] = round((index / len(videos)) * 100, 1)
                            self._write_status(status)
                            progress.update(ai_clipping_task, visible=False)
                            progress.advance(videos_task)
                            progress.update(
                                videos_task,
                                description=f"[cyan]Videos: {index}/{len(videos)}",
                            )
    
        finally:
            self._offload_runtime_resources(whisper_model=whisper_model)

        status["current_step"] = "completed"
        status["current_video"] = None
        status["progress_percent"] = 100.0
        self._write_status(status)
        summary = Table(show_header=False, box=None, pad_edge=False)
        summary.add_column("Key", style="cyan", no_wrap=True)
        summary.add_column("Value", style="white")
        summary.add_row("AI clipping base chunks", str(run_stats["ai_clipping_base_chunks"]))
        summary.add_row("AI clipping bridge chunks", str(run_stats["ai_clipping_bridge_chunks"]))
        summary.add_row("AI clipping chunks scanned", str(run_stats["ai_clipping_chunks_scanned"]))
        summary.add_row("AI clips found", str(run_stats["ai_clipping_candidates"]))
        summary.add_row("Candidates after merge", str(run_stats["merged_candidates"]))
        summary.add_row("Clips extracted", str(run_stats["clips_extracted"]))
        summary.add_row("Output folder", str(self.output_dir))
        self.console.print(Panel(summary, title="Run Summary", border_style="green"))
        self.console.print("[bold green]Engine run completed.[/bold green]")
        self.logger.info("Clipping engine run finished")

