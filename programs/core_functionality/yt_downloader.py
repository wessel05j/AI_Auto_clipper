def yt_downloader(url: str, output: str):
    import yt_dlp
    import os

    os.makedirs(output, exist_ok=True)

    ydl_opts = {
        "format": "bv*+ba/b",            # best video + best audio (or best fallback)
        "merge_output_format": "mp4",    # final file format
        "outtmpl": f"{output}/%(title).200B.%(ext)s",
        "quiet": False,
        "noprogress": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"\nDownloading: {url}")
        ydl.download([url])
