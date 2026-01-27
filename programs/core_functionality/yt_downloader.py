def yt_downloader(url: str, output: str):
    import yt_dlp
    import os
    import time

    os.makedirs(output, exist_ok=True)

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
            'http': lambda n: min(2 ** n, 300),  # Exponential backoff up to 5 minutes
            'fragment': lambda n: min(2 ** n, 300),
        },
    }

    while True:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            break  # Success, exit loop
        except Exception as e:
            print(f"Download failed: {e}. Retrying in 30 seconds...")
            time.sleep(30)
