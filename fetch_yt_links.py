import yt_dlp
from datetime import datetime, timedelta
import os
from programs.components.load import load
from programs.components.write import write



def main()  :

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SYSTEM_DIR = os.path.join(BASE_DIR, "system")
    SETTINGS_FILE = os.path.join(SYSTEM_DIR, "settings.json")
    try:
        settings = load(SETTINGS_FILE)
    except FileNotFoundError:
        print("Settings file not found. Please run 'python settings.py' first to configure the application.")
        return

    channels = settings["channels"]
    channels_hours_limit = settings["channels_hours_limit"]

    def fetch_videos(channels=channels, hours_limit=channels_hours_limit):
        threshold = datetime.now() - timedelta(hours=hours_limit)
        links_found = []

        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'extract_flat': True,   # fast metadata only
            'playlistend': 15,      # check top N newest entries
            'ignoreerrors': True,
            'no_warnings': True,
            'socket_timeout': 30,  # timeout for network operations
        }

        print(f"Searching for videos uploaded after: {threshold.strftime('%Y-%m-%d %H:%M:%S')}")

        def parse_time(entry):
            ts = entry.get('timestamp')
            ud = entry.get('upload_date')
            if ts:
                return datetime.fromtimestamp(ts)
            if ud:
                return datetime.strptime(ud, '%Y%m%d')
            return None

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for channel_url in channels:
                print(f"Checking: {channel_url}")
                result = ydl.extract_info(channel_url, download=False)

                if not result or 'entries' not in result:
                    continue

                for entry in result['entries']:
                    if not entry:
                        continue

                    video_time = parse_time(entry)

                    # If flat data lacks time, fetch lightweight info for this item only
                    if not video_time:
                        try:
                            info = ydl.extract_info(entry.get('url'), download=False)
                            video_time = parse_time(info or {})
                        except Exception:
                            video_time = None

                    if not video_time:
                        continue  # skip items without reliable time

                    if video_time < threshold:
                        break  # entries are newest first; stop when older

                    links_found.append(entry.get('webpage_url') or entry.get('url'))
            
            return links_found
    try:
        print("Fetching recent Youtube videos...")
        links = fetch_videos()
        print(f"{len(links)} found")
        current_links = settings["youtube_list"]
        if links:
            for link in links:
                if link not in current_links:
                    current_links.append(link)
        settings["youtube_list"] = current_links
        print("Updating settings file with new links...")
        write(SETTINGS_FILE, settings)
    except Exception as e:
        print(f"Error fetching recent videos: {e}")
        return

if __name__ == "__main__":
    main()
    input("Press Enter to exit...")