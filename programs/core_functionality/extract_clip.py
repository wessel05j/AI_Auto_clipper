def extract_clip(clip: list, video: str, output: str, input: str, id: int):
    from moviepy.editor import VideoFileClip

    start_time, end_time = clip
    
    with VideoFileClip(video) as main_video:
        subclip = main_video.subclip(start_time, end_time)

        # Single output filename per call
        output_filename = f"{output}{video[len(input):-4]}{id}.mp4"

        subclip.write_videofile(output_filename)