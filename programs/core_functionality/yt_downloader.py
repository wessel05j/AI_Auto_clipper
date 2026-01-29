def yt_downloader(url: str, output: str):
    import yt_dlp
    import os
    import time
    import logging
    import random

    # Random User-Agents to avoid detection
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    ]

    os.makedirs(output, exist_ok=True)

    # Random delay before download (30-90 seconds)
    delay = random.randint(30, 90)
    logging.info(f"Waiting {delay}s before downloading...")
    time.sleep(delay)

    ydl_opts = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": f"{output}/%(title).200B.%(ext)s",
        "quiet": False,
        "noprogress": False,
        "socket_timeout": 60,
        "retries": 10,
        "fragment_retries": 10,
        "retry_sleep": 5,
        "retry_sleep_functions": {
            'http': lambda n: min(2 ** n, 300),
            'fragment': lambda n: min(2 ** n, 300),
        },
        "http_headers": {"User-Agent": random.choice(USER_AGENTS)},
    }

    while True:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            logging.info(f"Downloaded: {url}")
            break  # Success, exit loop
        except Exception as e:
            logging.warning(f"Download failed: {e}. Retrying in 30 seconds...")
            time.sleep(30)
