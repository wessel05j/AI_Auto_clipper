

def sanitize_filename(filename: str) -> str:
    import re
    # Remove or replace problematic characters: < > : " / \ | ? * '
    sanitized = re.sub(r'[<>:"/\\|?*\']', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    return sanitized



def extract_clip(clip: list, video: str, output: str, input: str, id: int):
    from moviepy.editor import VideoFileClip

    start_time, end_time = clip
    
    with VideoFileClip(video) as main_video:
        subclip = main_video.subclip(start_time, end_time)

        video_sanitized = sanitize_filename(video)
        # Single output filename per call
        output_filename = f"{output}{video_sanitized[len(input):-4]}{id}.mp4"

        subclip.write_videofile(output_filename)