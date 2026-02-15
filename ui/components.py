from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from utils.validators import APP_GITHUB, APP_NAME, APP_VERSION


ASCII_LOGO = r"""
    _    ___      _   _   _ _____ ___      ____ _     ___ ____  ____  _____ ____
   / \  |_ _|    / \ | | | |_   _/ _ \    / ___| |   |_ _|  _ \|  _ \| ____|  _ \
  / _ \  | |    / _ \| | | | | || | | |  | |   | |    | || |_) | |_) |  _| | |_) |
 / ___ \ | |   / ___ \ |_| | | || |_| |  | |___| |___ | ||  __/|  __/| |___|  _ <
/_/   \_\___| /_/   \_\___/  |_| \___/    \____|_____|___|_|   |_|   |_____|_| \_\
"""


def build_header_panel() -> Panel:
    title = Text(ASCII_LOGO, style="bold cyan")
    subtitle = Text(f"{APP_NAME.upper()} v{APP_VERSION}", style="bold white")
    github = Text(APP_GITHUB, style="bright_blue")
    body = Align.center(Text.assemble(title, "\n", subtitle, "\n", github))
    return Panel(body, border_style="cyan", padding=(1, 2))


def show_header(console: Console) -> None:
    console.print(build_header_panel())


def hard_clear(console: Console) -> None:
    """
    Clear visible terminal and request scrollback purge (best-effort).
    """
    if os.name == "nt":
        try:
            os.system("cls")
        except Exception:
            pass
    if sys.stdout.isatty():
        try:
            # 3J clears scrollback on terminals that support ANSI VT sequences.
            sys.stdout.write("\033[3J\033[2J\033[H")
            sys.stdout.flush()
        except Exception:
            pass
    console.clear(home=True)


def startup_animation(console: Console) -> None:
    steps = [
        "Loading terminal UI",
        "Checking local runtime",
        "Preparing AI components",
    ]
    with console.status("[bold cyan]Booting AI Auto Clipper...", spinner="dots"):
        for _ in steps:
            time.sleep(0.22)


def warning_panel(message: str) -> Panel:
    return Panel(
        Text(message, style="bold yellow"),
        title="Warning",
        border_style="yellow",
    )


def success_panel(message: str) -> Panel:
    return Panel(
        Text(message, style="bold green"),
        title="Ready",
        border_style="green",
    )


def edit_text_in_editor(initial_text: str, filename_hint: str = "system_prompt.txt") -> str:
    """
    Open text in external editor for full navigation/editing.
    Falls back to returning the original text when editor launch fails.
    """
    temp_dir = Path(tempfile.gettempdir())
    temp_file = temp_dir / filename_hint
    temp_file.write_text(initial_text, encoding="utf-8")

    editor = os.environ.get("EDITOR", "").strip()
    try:
        if os.name == "nt":
            command = [editor, str(temp_file)] if editor else ["notepad", str(temp_file)]
            subprocess.run(command, check=False)
        else:
            if editor:
                subprocess.run([editor, str(temp_file)], check=False)
            else:
                subprocess.run(["nano", str(temp_file)], check=False)
    except Exception:
        return initial_text

    try:
        edited = temp_file.read_text(encoding="utf-8")
    except Exception:
        return initial_text

    cleaned = edited.strip()
    if not cleaned:
        return initial_text
    return edited
