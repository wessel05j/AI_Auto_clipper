import logging
import os
import subprocess


def ollama_off() -> None:
    """
    Best-effort shutdown for local Ollama processes.
    Windows uses taskkill; Unix uses pkill if available.
    """
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/F", "/T", "/IM", "ollama*"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logging.debug("Successfully terminated Ollama processes.")
            else:
                logging.debug("Ollama was not running.")
            return

        result = subprocess.run(
            ["pkill", "-f", "ollama"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logging.debug("Successfully terminated Ollama processes.")
        else:
            logging.debug("Ollama was not running or pkill not available.")
    except Exception as e:
        logging.error(f"An error occurred while trying to terminate Ollama: {e}")
