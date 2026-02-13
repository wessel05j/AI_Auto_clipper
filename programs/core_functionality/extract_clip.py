def sanitize_filename(filename: str) -> str:
    import re
    # Remove or replace problematic characters: < > : " / \ | ? * '
    sanitized = re.sub(r'[<>:"/\\|?*\']', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    return sanitized



def extract_clip(
    clip: list,
    video: str,
    output: str,
    id: int,
    rating: float,
    main_video=None,
) -> None:
    from moviepy.editor import VideoFileClip
    import os

    def _write_clip(video_handle):
        start_time, end_time = clip[0], clip[1]

        # Clamp end_time to stay within video duration (10ms buffer to avoid edge issues)
        video_duration = video_handle.duration
        clip_end = min(end_time, video_duration)

        subclip = video_handle.subclip(start_time, clip_end)

        video_basename = os.path.splitext(os.path.basename(video))[0]
        video_sanitized = sanitize_filename(video_basename)
        output_filename = os.path.join(output, f"{video_sanitized}{id}r{rating}.mp4")
        try:
            subclip.write_videofile(
                output_filename, 
                temp_audiofile=os.path.join(output, f"temp_audio_{id}.mp4"), # Changed to mp4/m4a
                remove_temp=True,     # Ensures the temp file is deleted immediately after
                codec="libx264",      # Explicitly set codec
                audio_codec="aac",    # Explicitly set audio codec
                threads=4             # Limit threads to prevent memory spikes
            )
        except Exception as e:
            clip_end = min(end_time + 0.1, video_duration - 0.1)  # Further reduce by 100ms and add buffer
            subclip = video_handle.subclip(start_time, clip_end)
            subclip.write_videofile(
                output_filename, 
                temp_audiofile=os.path.join(output, f"temp_audio_{id}.mp4"), # Changed to mp4/m4a
                remove_temp=True,     # Ensures the temp file is deleted immediately after
                codec="libx264",      # Explicitly set codec
                audio_codec="aac",    # Explicitly set audio codec
                threads=4             # Limit threads to prevent memory spikes
            )

    if main_video is None:
        with VideoFileClip(video) as opened_video:
            _write_clip(opened_video)
    else:
        _write_clip(main_video)
