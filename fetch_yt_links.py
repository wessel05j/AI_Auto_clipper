from datetime import datetime, timedelta
import os
from programs.components.load import load
from programs.components.write import write
from programs.core_functionality.yt_service import YTService



def main():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SYSTEM_DIR = os.path.join(BASE_DIR, "system")
    SETTINGS_FILE = os.path.join(SYSTEM_DIR, "settings.json")
    try:
        settings = load(SETTINGS_FILE)
    except FileNotFoundError:
        print("Settings file not found. Please run 'python settings.py' first to configure the application.")
        return

    channels = settings.get("channels", [])
    channels_hours_limit = settings.get("channels_hours_limit", 24)
    youtube_client = YTService(base_dir=BASE_DIR)

    def fetch_videos(channels=channels, hours_limit=channels_hours_limit):
        threshold = datetime.now() - timedelta(hours=hours_limit)
        print(f"Searching for videos uploaded after: {threshold.strftime('%Y-%m-%d %H:%M:%S')}")
        if not channels:
            return []
        return youtube_client.fetch_recent_links(
            channels=channels,
            hours_limit=hours_limit,
            playlistend=15,
        )

    try:
        print("Fetching recent Youtube videos...")
        links = fetch_videos()
        print(f"{len(links)} found")
        current_links = settings.get("youtube_list", [])
        added = 0
        if links:
            for link in links:
                if link not in current_links:
                    current_links.append(link)
                    added += 1
        settings["youtube_list"] = current_links
        print(f"Updating settings file with {added} new links...")
        write(SETTINGS_FILE, settings)
    except Exception as e:
        print(f"Error fetching recent videos: {e}")
        return

if __name__ == "__main__":
    main()
    input("Press Enter to exit...")
