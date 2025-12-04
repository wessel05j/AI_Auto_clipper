def yt_downloader(url: str, output: str):
    import yt_dlp
    import os

    os.makedirs(output, exist_ok=True)

    ydl_opts = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": f"{output}/%(title).200B.%(ext)s",
        "quiet": False,
        "noprogress": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"\nDownloading: {url}")
        try:
            ydl.download([url])
            return True
        except yt_dlp.utils.DownloadError as e:
            print(f"Download failed for {url}: {e}")
            return False
