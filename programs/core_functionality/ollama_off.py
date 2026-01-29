import subprocess
import logging

def ollama_off():
    """
    Forcefully terminates all Ollama processes and sub-processes 
    to ensure 100% VRAM and RAM recovery.
    """
    try:
        # /F = Forcefully terminate the process
        # /T = Terminates child processes as well (the actual model runners)
        # /IM = Image Name (targets the process by name)
        # Using 'ollama*' covers ollama.exe and any internal runner engines
        result = subprocess.run(
            ["taskkill", "/F", "/T", "/IM", "ollama*"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logging.debug("Successfully terminated Ollama. Memory has been reclaimed.")
        else:
            logging.debug("Ollama was not running.")
            
    except Exception as e:
        logging.error(f"An error occurred while trying to kill the process: {e}")