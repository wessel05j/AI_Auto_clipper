import os
import time
import logging
from tqdm import tqdm
# Core files
from programs.core_functionality.yt_downloader import yt_downloader
from programs.core_functionality.transcribing import transcribe_video
from programs.core_functionality.chunking import chunking
from programs.core_functionality.merge_segments import merge_segments
from programs.core_functionality.extract_clip import extract_clip
from programs.core_functionality.ollama_on import ollama_on
from programs.core_functionality.ollama_off import ollama_off
from programs.core_functionality.ollama_chat import ollama_chat
from programs.core_functionality.ollama_scanning import ollama_scanning

# Components
from programs.components.file_exists import file_exists
from programs.components.load import load
from programs.components.write import write
from programs.components.scan_videos import scan_videos

def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

def init():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SYSTEM_DIR = os.path.join(BASE_DIR, "system")
    if not os.path.exists(SYSTEM_DIR):
        os.makedirs(SYSTEM_DIR)
    
    # Setup logging
    log_file = os.path.join(SYSTEM_DIR, "log.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Clear old logs if file > 1MB
    if os.path.exists(log_file) and os.path.getsize(log_file) > 1024 * 1024:
        with open(log_file, 'w') as f:
            f.write("")  # Clear file
        logging.info("Log file cleared due to size > 1MB")

    # Clear oldest videos in temp if > 10MB
    global TEMP_DIR
    TEMP_DIR = os.path.join(BASE_DIR, "temp")
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    total_size = 0
    file_list = []
    for root, dirs, files in os.walk(TEMP_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            file_list.append((file_path, file_size, os.path.getctime(file_path)))
    if total_size > 10 * 1024 * 1024:  # 10MB
        # Sort by creation time (oldest first)
        file_list.sort(key=lambda x: x[2])
        size_freed = 0
        for file_path, file_size, _ in file_list:
            os.remove(file_path)
            size_freed += file_size
            logging.info(f"Deleted temp file: {file_path} to free up space.")
            if total_size - size_freed <= 10 * 1024 * 1024:
                break
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
    
    global STATUS_FILE
    STATUS_FILE = os.path.join(SYSTEM_DIR, "status.json")
    
    global settings, model, transcribing_model, user_query, youtube_list, merge_distance, max_token, system_message, ai_loops, ollama_url, max_ai_tokens, temperature
    try:
        settings = load(SETTINGS_FILE)
    except FileNotFoundError:
        print("Settings file not found. Please run 'python settings.py' first to configure the application.")
        input("Press Enter to exit...")
        exit(1)
        
    model = settings["ai_model"]
    transcribing_model = settings["transcribing_model"]
    user_query = settings["user_query"]
    system_message = settings["system_query"]
    max_token = settings["max_tokens"]
    max_ai_tokens = settings["max_ai_tokens"]
    youtube_list = settings["youtube_list"]
    merge_distance = settings["merge_distance"]
    ai_loops = settings["ai_loops"]
    ollama_url = settings["ollama_url"]
    temperature = settings["temperature"]
    rerun_temp_files = settings["rerun_temp_files"]

    if os.path.exists(TRANSCRIBING_FILE) or os.path.exists(AI_FILE) or os.path.exists(CLIPS_FILE):
        print("Warning: Previous session files detected.")
        if rerun_temp_files:
            print("Rerun Temp Files is enabled. Previous session files will be reused.")
        else:
            print("Rerun Temp Files is disabled. Previous session files will be deleted.")
            if os.path.exists(TRANSCRIBING_FILE):
                os.remove(TRANSCRIBING_FILE)
            if os.path.exists(AI_FILE):
                os.remove(AI_FILE)
            if os.path.exists(CLIPS_FILE):
                os.remove(CLIPS_FILE)  

def start() -> None:
    #Booting procedure
    print("Botting up")
    try:
        print("Loading transcription model...")
        import whisper
        import torch
        if torch.cuda.is_available():
            print("CUDA device found. Using GPU for transcription.")
            device = "cuda"
        else:
            print("No CUDA device found. Using CPU for transcription.")
            device = "cpu"
        model_whisper = whisper.load_model(transcribing_model, device=device)
        #checking ollama
        print("Turning ollama on...")
        ollama_on(ollama_url)
        print("Trying to communicate with ollama...")
        response = ollama_chat(
            model=model,
            prompt="This is just a connectivity test.",
            system_message="Answer briefly with 'Ollama is connected.'",
            temperature=1.0,
            think="low",
            stream=False,
            max_tokens=50,
            url=ollama_url,
        )
        print(response)
        print("Ollama communication successful.")
        print("Ollama will stay on for processing.")
        print("Running system in")
        print("3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)
        cls()
    except Exception as e:
        print(f"Error during booting procedure: {e}")
        return

    #--------------------------------------------------------------------------------#
    # Youtube Downloading
    #--------------------------------------------------------------------------------#
    print(f"Downloading: {len(youtube_list)} youtube links...")
    try:
        while len(youtube_list) > 0:
                for link in list(youtube_list):
                    yt_downloader(link, INPUT_DIR)  
                    youtube_list.remove(link)
                    settings = load(SETTINGS_FILE)
                    settings["youtube_list"] = youtube_list
                    write(SETTINGS_FILE, settings)
        cls()
    except Exception as e:
        print(f"Error downloading from YouTube: {e}")
        return 
    #--------------------------------------------------------------------------------#
    # Youtube Downloading
    #--------------------------------------------------------------------------------#

    videos = scan_videos(INPUT_DIR)
    # Initialize status
    status = {
        "total_videos": len(videos),
        "progress": 0,
        "current_video": "None",
        "current_step": "Starting"
    }
    write(STATUS_FILE, status)
    
    for i, video in enumerate(tqdm(videos, desc="Processing videos", unit="video")):
        status["current_video"] = os.path.basename(video)
        status["progress"] = (i / len(videos)) * 100
        status["current_step"] = "Transcribing"
        write(STATUS_FILE, status)
        #--------------------------------------------------------------------------------#
        # Transcribing
        #--------------------------------------------------------------------------------#
        print("Transcribing...")
        try:
            if file_exists(TRANSCRIBING_FILE):
                transcribed_text = load(TRANSCRIBING_FILE)
            else:
                transcribed_text = transcribe_video(video, model_whisper)
                write(TRANSCRIBING_FILE, transcribed_text)
        except Exception as e:
            logging.error(f"Error during transcribing for video {video}: {e}")
            continue
        #--------------------------------------------------------------------------------#
        # Transcribing
        #--------------------------------------------------------------------------------#


        #--------------------------------------------------------------------------------#
        # Chunking
        #--------------------------------------------------------------------------------#
        print("Chunking...")
        status["current_step"] = "Chunking"
        write(STATUS_FILE, status)
        try:
            chunked_transcribed_text = chunking(transcribed_text, max_token)
            print(f"Chunks created: {len(chunked_transcribed_text)}") 
        except Exception as e:
            logging.error(f"Error during chunking for video {video}: {e}")
            continue
        #--------------------------------------------------------------------------------#
        # Chunking
        #--------------------------------------------------------------------------------#


        #--------------------------------------------------------------------------------#
        # AI scanning
        #--------------------------------------------------------------------------------#
        print("AI Scanning...")
        status["current_step"] = "AI Scanning"
        write(STATUS_FILE, status)
        try:
            if file_exists(AI_FILE):
                AI_output = load(AI_FILE)
            else:
                AI_output = []
                for chunked in tqdm(chunked_transcribed_text, desc="Scanning chunks", unit="chunk", leave=False):
                    text = ""
                    for segment in chunked:
                        tokens_segment_text = f"{segment[0]} {segment[1]} {segment[2]}"
                        text += tokens_segment_text + "\n"
                    combined_outputs = []
                    for _ in range(ai_loops):
                        output = ollama_scanning(
                            chunked,
                            user_query,
                            model,
                            chunked_transcribed_text,
                            system_message,
                            temperature=temperature,
                            max_tokens=max_ai_tokens,
                            url=ollama_url
                        )
                        if isinstance(output, list) and output:
                            combined_outputs.extend(output)
                    AI_output.append(combined_outputs)
                write(AI_FILE, AI_output)
        except Exception as e:
            logging.error(f"Error during AI scanning for video {video}: {e}")
            continue 
        #--------------------------------------------------------------------------------#
        # AI scanning
        #--------------------------------------------------------------------------------#


        #--------------------------------------------------------------------------------#
        # Segment Cleanup
        #--------------------------------------------------------------------------------#
        print("Segment Cleanup...")
        status["current_step"] = "Merging Segments"
        write(STATUS_FILE, status)
        try:
            list_of_clips = merge_segments(AI_output, merge_distance)
        except Exception as e:
            logging.error(f"Error during segment cleanup for video {video}: {e}")
            continue 
        #--------------------------------------------------------------------------------#
        # Segment Cleanup
        #--------------------------------------------------------------------------------#


        #--------------------------------------------------------------------------------#
        # Video Clipping
        #--------------------------------------------------------------------------------#
        print("Video Clipping...")
        status["current_step"] = "Extracting Clips"
        write(STATUS_FILE, status)
        try:
            if file_exists(CLIPS_FILE):
                list_of_clips = load(CLIPS_FILE)

            print(f"Clips to extract: {len(list_of_clips)}")
            for clip in list(list_of_clips):
                extract_clip(clip, video, OUTPUT_DIR, len(list_of_clips), clip[2])
                list_of_clips.remove(clip)
                write(CLIPS_FILE, list_of_clips)
        except Exception as e:
            logging.error(f"Error during video clipping for video {video}: {e}")
            continue 
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
                base_name = os.path.basename(video)
                temp_video_path = os.path.join(TEMP_DIR, base_name)
                os.rename(video, temp_video_path)
        except Exception as e:
            print(f"Error during system cleanup for video {video}: {e}")
            return 
        #--------------------------------------------------------------------------------#
        # System Cleanup
        #--------------------------------------------------------------------------------#

    # Update status to completed
    status["current_step"] = "Completed"
    status["progress"] = 100
    write(STATUS_FILE, status)

    # Turn off Ollama after all processing
    print("Turning Ollama off...")
    ollama_off()
    print("Ollama is off.")

if __name__ == "__main__":
    init()
    start()
    input("Press Enter to exit...")
