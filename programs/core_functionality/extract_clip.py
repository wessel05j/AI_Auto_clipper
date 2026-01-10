

def sanitize_filename(filename: str) -> str:
    import re
    # Remove or replace problematic characters: < > : " / \ | ? * '
    sanitized = re.sub(r'[<>:"/\\|?*\']', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    return sanitized



def extract_clip(clip: list, video: str, output: str, input: str, id: int, rating: float) -> None:
    from moviepy.editor import VideoFileClip
    import os

    start_time, end_time = clip[0], clip[1]
    
    with VideoFileClip(video) as main_video:
        subclip = main_video.subclip(start_time, end_time)

        video_basename = os.path.splitext(os.path.basename(video))[0]
        video_sanitized = sanitize_filename(video_basename)
        output_filename = os.path.join(output, f"{video_sanitized}{id}r{rating}.mp4")

        subclip.write_videofile(output_filename, temp_audiofile=os.path.join(output, f"temp_audio_{id}.mp3"))