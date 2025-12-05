import os
import traceback

# Core files
from programs.core_functionality.scan_videos import scan_videos
from programs.core_functionality.yt_downloader import yt_downloader
from programs.core_functionality.transcribing import transcribe_video
from programs.core_functionality.chunking import chunking
from programs.core_functionality.ai_scanning import ai_clipping
from programs.core_functionality.merge_segments import merge_segments
from programs.core_functionality.extract_clip import extract_clip

# Components
from programs.components.file_exists import file_exists
from programs.components.load import load
from programs.components.wright import wright

# Setup stage
from programs.setup_stage.setup_stage import setup_stage


settings_path = "system/settings.json"


def log_fatal_error(message: str, exc: Exception) -> None:
    """Print a clear fatal error with traceback so the user knows exactly what happened."""
    print("\n==== FATAL ERROR ====")
    print(message)
    print("Error type:", type(exc).__name__)
    print("Error message:", exc)
    print("Traceback:")
    traceback.print_exc()
    print("====================\n")


def main() -> None:
    try:
        run = setup_stage()
    except Exception as e:
        log_fatal_error("Fatal error during setup stage.", e)
        return

    if not run:
        print("Setup stage failed, exiting main loop...")
        return

    #Settings from setup stage
    settings = load(settings_path)
    clips_input = settings["setup_variables"]["input_folder"]
    clips_output = settings["setup_variables"]["output_folder"]
    base_url = settings["setup_variables"]["base_url"]
    model = settings["setup_variables"]["ai_model"]
    transcribing_model = settings["setup_variables"]["transcribing_model"]
    user_query = settings["setup_variables"]["user_query"]
    youtube_list = settings["setup_variables"]["youtube_list"]
    merge_distance = settings["setup_variables"]["merge_distance"]
    max_token = settings["setup_variables"]["max_tokens"]
    system_message_not_chunked = settings["system_variables"]["AI_instructions"]
    system_message_chunked = settings["system_variables"]["AI_instructions_w_chunking"]

    transcribing_name = settings["system_variables"]["transcribing_name"]
    AI_name = settings["system_variables"]["AI_name"]
    clips_name = settings["system_variables"]["clips_name"]

    #Download youtube videos if the user has declared them:
    if len(youtube_list) > 0:
        #If list is more than 10 videos, only download 10 at a time and then wait 10 minutes before downloading more
        #This is to avoid getting blocked by youtube for too many requests
        print(f"Downloading {len(youtube_list)} youtube videos...")
        try:
            if len(youtube_list) > 10:
                # Work on a copy so we don't modify the list while iterating
                for link in youtube_list[:10]:
                    try:
                        yt_downloader(link, clips_input)  # Downloading the videos to input folder
                    except Exception as e:
                        print(f"Error downloading YouTube video {link}:", e)
                        # Stop the whole run on hard failure instead of silently continuing
                        raise
                    youtube_list.remove(link)
                    #Update youtube_list in settings:
                    settings = load(settings_path)
                    settings["setup_variables"]["youtube_list"] = youtube_list
                    wright(settings_path, settings)
                print("Waiting 10 minutes before downloading more videos...")
                import time
                time.sleep(600)  # Wait 10 minutes
            else:
                for link in list(youtube_list):
                    try:
                        yt_downloader(link, clips_input)  # Downloading the videos to input folder
                    except Exception as e:
                        print(f"Error downloading YouTube video {link}:", e)
                        raise
                    youtube_list.remove(link)
                    #Update youtube_list in settings:
                    settings = load(settings_path)
                    settings["setup_variables"]["youtube_list"] = youtube_list
                    wright(settings_path, settings)
        except Exception as e:
            log_fatal_error("Unexpected error in YouTube download loop.", e)
            return
        

    
    #Collection all videos for clipping:
    print("Collection videos...")
    videos = scan_videos(clips_input)
    iterated = 0
    print(f"System is 0% finished")

    for video in videos:
        print(video)
        videos_update = scan_videos(clips_input)
        print(f"Videos left: {len(videos_update)}")
        print("AI_clipper: ", video)

        try:
            #Transcribing:
            if file_exists(f"system/{transcribing_name}"):
                print("Already transcribed...")
                transcribed_text = load(f"system/{transcribing_name}")
            else:
                print("Transcribing...")
                transcribed_text = transcribe_video(video, transcribing_model)
                wright(f"system/{transcribing_name}", transcribed_text)
        except Exception as e:
            log_fatal_error(f"Error during transcription for video {video}.", e)
            return

        try:
            #Chunking
            print("Chunking...")
            chunked_transcribed_text = chunking(transcribed_text, max_token, model)
            print(f"Chunks created: {len(chunked_transcribed_text)}")
        except Exception as e:
            log_fatal_error(f"Error during chunking for video {video}.", e)
            return

        try:
            #AI chosen clips:
            if file_exists(f"system/{AI_name}"):
                print("Already done AI scanning...")
                AI_output = load(f"system/{AI_name}")
            else:
                print("Scanning with AI...")
                AI_output = []
                for chunked in chunked_transcribed_text:
                    try:
                        output = ai_clipping(
                            chunked,
                            user_query,
                            base_url,
                            model,
                            chunked_transcribed_text,
                            system_message_not_chunked,
                            system_message_chunked,
                        )
                    except Exception as e:
                        log_fatal_error(
                            "AI model failed to process the chunked transcribed text. This may be due to exceeding token limits or a connectivity issue.",
                            e,
                        )
                        return
                    # Append a fresh copy to avoid any unintended shared mutations
                    AI_output.append(output)

                if not AI_output:
                    print("No AI output generated; skipping this video.")
                    continue

                wright(f"system/{AI_name}", AI_output)
        except Exception as e:
            log_fatal_error(f"Unexpected error during AI scanning for video {video}.", e)
            return

        try:
            #Segment Cleanup
            print("Finding AI scanning in transcribed text...")
            list_of_clips = merge_segments(AI_output, merge_distance)
            print(f"Found: {len(list_of_clips)} Clips!")
        except Exception as e:
            log_fatal_error(f"Error during segment merging for video {video}.", e)
            return

        #Video clipping
        if file_exists(f"system/{clips_name}"):
            print("Continuing Clipping...")
            list_of_clips = load(f"system/{clips_name}")
        else:
            print("Starting Clipping...")

        for clip in list(list_of_clips):
            print(f"Clips left: {len(list_of_clips)}")
            try:
                filename = extract_clip(clip, video, clips_output, clips_input, len(list_of_clips))
                list_of_clips.remove(clip)
                wright(f"system/{clips_name}", list_of_clips)
            except Exception as e:
                log_fatal_error(f"Error extracting clip {clip} from video {video}.", e)
                return

        #System updating
        iterated += 1
        print(f"System is {iterated/len(videos)*100:.0f}% finished!")
        try:
            if os.path.exists(f"system/{AI_name}"):
                os.remove(f"system/{AI_name}")
            if os.path.exists(f"system/{transcribing_name}"):
                os.remove(f"system/{transcribing_name}")
            if os.path.exists(f"system/{clips_name}"):
                os.remove(f"system/{clips_name}")
            if os.path.exists(video):
                os.remove(video)
        except Exception as e:
            log_fatal_error(f"Error during cleanup for video {video}.", e)
            return

    #Shut down after all videos have been iterated
    print("System finished!")


if __name__ == "__main__":
    main()
