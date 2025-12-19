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

def terminal_log(videos_amount: int, current_videos_amount: int, video_name: str, youtube_amount=None, current_youtube_amount=None, youtube_stage=False, transcribing_stage=False, ai_stage=False, clipping_stage=False):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("======System Log======")
    if youtube_amount is not None and current_youtube_amount is not None:
        print(f"Youtube Videos: {current_youtube_amount}/{youtube_amount} ({(current_youtube_amount/youtube_amount)*100:.2f}%)")
    print(f"Vidoes: {current_videos_amount}/{videos_amount} ({(current_videos_amount/videos_amount)*100:.2f}%)")
    print(f"Current Video: {video_name}")
    if youtube_stage:
        print("Downloading Youtube Video...")
    if transcribing_stage:
        print("Transcribing Video...")
    if ai_stage:
        print("AI Scanning Video...")
    if clipping_stage:
        print("Clipping Video...")
    print("======================")

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
    youtube_amount = len(youtube_list)
    while len(youtube_list) > 0:
        terminal_log(videos_amount=None, current_videos_amount=None, video_name="", youtube_amount=youtube_amount, current_youtube_amount=len(youtube_list), youtube_stage=True)
        try:
            if len(youtube_list) > 10:
                for link in youtube_list[:10]:
                    terminal_log(videos_amount=None, current_videos_amount=None, video_name="", youtube_amount=youtube_amount, current_youtube_amount=len(youtube_list), youtube_stage=True)
                    yt_downloader(link, INPUT_DIR)  
                    youtube_list.remove(link)
                    settings = load(SETTINGS_FILE)
                    settings["setup_variables"]["youtube_list"] = youtube_list
                    wright(SETTINGS_FILE, settings)
                import time
                time.sleep(600)  
            else:
                for link in list(youtube_list):
                    terminal_log(videos_amount=None, current_videos_amount=None, video_name="", youtube_amount=youtube_amount, current_youtube_amount=len(youtube_list), youtube_stage=True)
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
    VIDEO_AMOUNT = len(videos)

    for video in videos:
        videos_update = scan_videos(INPUT_DIR)
        terminal_log(videos_amount=VIDEO_AMOUNT, current_videos_amount=videos_update, video_name=video)


        #--------------------------------------------------------------------------------#
        # Transcribing
        #--------------------------------------------------------------------------------#
        terminal_log(videos_amount=VIDEO_AMOUNT, current_videos_amount=videos_update, video_name=video, transcribing_stage=True)
        try:
            if file_exists(TRANSCRIBING_FILE):
                transcribed_text = load(TRANSCRIBING_FILE)
            else:
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
        terminal_log(videos_amount=VIDEO_AMOUNT, current_videos_amount=videos_update, video_name=video, ai_stage=True)
        try:
            if file_exists(AI_FILE):
                AI_output = load(AI_FILE)
            else:
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
        terminal_log(videos_amount=VIDEO_AMOUNT, current_videos_amount=videos_update, video_name=video, clipping_stage=True)
        try:
            if file_exists(CLIPS_FILE):
                list_of_clips = load(CLIPS_FILE)

            for clip in list(list_of_clips):
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

if __name__ == "__main__":
    init()
    start()
