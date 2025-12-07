import os
import traceback

# Core files
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
from programs.components.scan_videos import scan_videos

# Setup stage
from programs.setup_stage.setup_stage import setup_stage

def log_fatal_error(message: str, exc: Exception) -> None:
    """Print a clear fatal error with traceback so the user knows exactly what happened."""
    print("\n==== FATAL ERROR ====")
    print(message)
    print("Error type:", type(exc).__name__)
    print("Error message:", exc)
    print("Traceback:")
    traceback.print_exc()
    print("====================\n")

def init():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SYSTEM_DIR = os.path.join(BASE_DIR, "system")
    if not os.path.exists(SYSTEM_DIR):
        os.makedirs(SYSTEM_DIR)
    global INPUT_DIR
    INPUT_DIR = os.path.join(BASE_DIR, "input")
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    global SETTINGS_FILE
    SETTINGS_FILE = os.path.join(SYSTEM_DIR, "settings.json")
    global TRANSCRIBING_FILE
    TRANSCRIBING_FILE = os.path.join(SYSTEM_DIR, "transcribing.json")
    global AI_FILE
    AI_FILE = os.path.join(SYSTEM_DIR, "AI.json")
    global CLIPS_FILE
    CLIPS_FILE = os.path.join(SYSTEM_DIR, "clips.json")

    setup_stage(SETTINGS_FILE)
    
    global settings, base_url, model, transcribing_model, user_query, youtube_list, merge_distance, max_token, system_message_not_chunked, system_message_chunked
    settings = load(SETTINGS_FILE)
    base_url = settings["setup_variables"]["base_url"]
    model = settings["setup_variables"]["ai_model"]
    transcribing_model = settings["setup_variables"]["transcribing_model"]
    user_query = settings["setup_variables"]["user_query"]
    youtube_list = settings["setup_variables"]["youtube_list"]
    merge_distance = settings["setup_variables"]["merge_distance"]
    max_token = settings["setup_variables"]["max_tokens"]
    system_message_not_chunked = settings["system_variables"]["AI_instructions"]
    system_message_chunked = settings["system_variables"]["AI_instructions_w_chunking"]

def start() -> None:
    #--------------------------------------------------------------------------------#
    # Youtube Downloading
    #--------------------------------------------------------------------------------#
    if len(youtube_list) > 0:
        print(f"Downloading {len(youtube_list)} youtube videos...\n")
        try:
            if len(youtube_list) > 10:
                for link in youtube_list[:10]:
                    yt_downloader(link, INPUT_DIR)  
                    youtube_list.remove(link)
                    settings = load(SETTINGS_FILE)
                    settings["setup_variables"]["youtube_list"] = youtube_list
                    wright(SETTINGS_FILE, settings)
                print("Waiting 10 minutes before downloading more videos...")
                import time
                time.sleep(600)  
            else:
                for link in list(youtube_list):
                    yt_downloader(link, INPUT_DIR)  
                    youtube_list.remove(link)
                    settings = load(SETTINGS_FILE)
                    settings["setup_variables"]["youtube_list"] = youtube_list
                    wright(SETTINGS_FILE, settings)
        except Exception as e:
            log_fatal_error("Unexpected error in YouTube download loop.", e)
            return
    print("Videos Completed!------------------------------------------------\n")
    #--------------------------------------------------------------------------------#
    # Youtube Downloading
    #--------------------------------------------------------------------------------#

    videos = scan_videos(INPUT_DIR)
    iterated = 0
    print(f"System is 0% finished")

    for video in videos:
        print(video)
        videos_update = scan_videos(INPUT_DIR)
        print(f"Videos left: {len(videos_update)}")
        print("Next: ", video)

        #--------------------------------------------------------------------------------#
        # Transcribing
        #--------------------------------------------------------------------------------#
        try:
            if file_exists(TRANSCRIBING_FILE):
                print("Already done Transcribing...")
                transcribed_text = load(TRANSCRIBING_FILE)
            else:
                print("Transcribing...")
                transcribed_text = transcribe_video(video, transcribing_model)
                wright(TRANSCRIBING_FILE, transcribed_text)
        except Exception as e:
            log_fatal_error(f"Error during transcription for video {video}.", e)
            return
        #--------------------------------------------------------------------------------#
        # Transcribing
        #--------------------------------------------------------------------------------#


        #--------------------------------------------------------------------------------#
        # Chunking
        #--------------------------------------------------------------------------------#
        try:
            print("Chunking...")
            chunked_transcribed_text = chunking(transcribed_text, max_token, model)
            print(f"Chunks created: {len(chunked_transcribed_text)}") 
        except Exception as e:
            log_fatal_error(f"Error during chunking for video {video}.", e)
            return
        #--------------------------------------------------------------------------------#
        # Chunking
        #--------------------------------------------------------------------------------#


        #--------------------------------------------------------------------------------#
        # AI scanning
        #--------------------------------------------------------------------------------#
        try:
            if file_exists(AI_FILE):
                print("Already done AI scanning...")
                AI_output = load(AI_FILE)
            else:
                print("AI scanning...")
                AI_output = []
                for chunked in chunked_transcribed_text:
                    output = ai_clipping(
                        chunked,
                        user_query,
                        base_url,
                        model,
                        chunked_transcribed_text,
                        system_message_not_chunked,
                        system_message_chunked,
                    )
                    AI_output.append(output)
                wright(AI_FILE, AI_output)
        except Exception as e:
            log_fatal_error(f"Unexpected error during AI scanning for video {video}.", e)
            return
        #--------------------------------------------------------------------------------#
        # AI scanning
        #--------------------------------------------------------------------------------#


        #--------------------------------------------------------------------------------#
        # Segment Cleanup
        #--------------------------------------------------------------------------------#
        try:
            print("Segment Cleanup...")
            list_of_clips = merge_segments(AI_output, merge_distance)
            print(f"Found: {len(list_of_clips)} Clips!")
        except Exception as e:
            log_fatal_error(f"Error during segment merging for video {video}.", e)
            return
        #--------------------------------------------------------------------------------#
        # Segment Cleanup
        #--------------------------------------------------------------------------------#


        #--------------------------------------------------------------------------------#
        # Video Clipping
        #--------------------------------------------------------------------------------#
        try:
            if file_exists(CLIPS_FILE):
                print("Continuing Video Clipping...")
                list_of_clips = load(CLIPS_FILE)
            else:
                print("Video Clipping...")

            for clip in list(list_of_clips):
                print(f"Clips left: {len(list_of_clips)}")
                extract_clip(clip, video, OUTPUT_DIR, INPUT_DIR, len(list_of_clips))
                list_of_clips.remove(clip)
                wright(CLIPS_FILE, list_of_clips)
        except Exception as e:
            log_fatal_error(f"Error during video clipping for video {video}.", e)
            return
        #--------------------------------------------------------------------------------#
        # Video Clipping
        #--------------------------------------------------------------------------------#


        #--------------------------------------------------------------------------------#
        # System Cleanup
        #--------------------------------------------------------------------------------#
        print(f"System is {iterated/len(videos)*100:.0f}% finished!")
        try:
            if os.path.exists(AI_FILE):
                os.remove(AI_FILE)
            if os.path.exists(TRANSCRIBING_FILE):
                os.remove(TRANSCRIBING_FILE)
            if os.path.exists(CLIPS_FILE):
                os.remove(CLIPS_FILE)
            if os.path.exists(video):
                os.remove(video)
        except Exception as e:
            log_fatal_error(f"Error during cleanup for video {video}.", e)
            return
        #--------------------------------------------------------------------------------#
        # System Cleanup
        #--------------------------------------------------------------------------------#
        iterated += 1
    print("System finished!")

if __name__ == "__main__":
    init()
    start()
