import subprocess
import time
import requests
import os
import logging

def ollama_on(url="http://localhost:11434/"):
    '''Wakes up the Ollama local server if it's not already running.
    Args:
        url (str): The base URL of the Ollama server.
    Returns:
        bool: True if Ollama is awake or successfully started, False otherwise.
    '''
    try:
        # Check if Ollama is already awake
        requests.get(url)
        logging.debug("Ollama is already running.")
    except requests.exceptions.ConnectionError:
        logging.info("Ollama not found. Waking it up...")
        
        # Set environment to reduce logging
        env = os.environ.copy()
        env['OLLAMA_DEBUG'] = 'ERROR'
        
        # Adjust command based on your Operating System
        if os.name == 'nt':  # Windows
            # Launches the Ollama app (assuming default install path)
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env, shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:  # macOS / Linux
            # 'ollama serve' starts the background daemon
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)
        
        # Wait for the server to initialize (usually 3-5 seconds)
        for i in range(10):
            try:
                requests.get(url)
                logging.info("Ollama is now awake and ready!")
                return True
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        
        logging.error("Failed to start Ollama. Please check your installation.")
        return False