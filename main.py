import os

#Core files
from programs.core_functionality.scan_videos import scan_videos
from programs.core_functionality.yt_downloader import yt_downloader
from programs.core_functionality.transcribing import transcribe_video
from programs.core_functionality.chunking import chunking
from programs.core_functionality.ai_scanning import ai_clipping
from programs.core_functionality.merge_segments import merge_segments
from programs.core_functionality.extract_clip import extract_clip

#Componenets
from programs.components.file_exists import file_exists
from programs.components.load import load
from programs.components.wright import wright

#Setup stage
from programs.setup_stage.setup_stage import setup_stage


settings_path = "system/settings.json"
run = setup_stage()
if run == False:
    print("Setup stage failed, exiting...")

while run:
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


    #Donwload youtube videos if the user have declared them:
    if len(youtube_list) > 0:
        #If list is more than 10 videos, only download 10 at a time and then wait 10 minutes before downloading more
        #This is to avoid getting blocked by youtube for too many requests
        print(f"Downloading youtube videos...{len(youtube_list)} videos to download.")
        if len(youtube_list) > 10:
            for link in youtube_list[:10]:
                yt_downloader(link, clips_input) #Downloading the videos to input folder
                youtube_list.remove(link)
                #Update youtube_list in settings:
                settings = load(settings_path)
                settings["setup_variables"]["youtube_list"] = youtube_list
                wright(settings_path, settings)
            print("Waiting 10 minutes before downloading more videos...")
            import time
            time.sleep(600) #Wait 10 minutes
    
    #Collection all videos for clipping:
    print("Collection videos...")
    videos = scan_videos(clips_input)
    iterated = 0
    print(f"System is 0% finished")
    for video in videos: 
        videos_update = scan_videos(clips_input)
        print(f"Videos left: {len(videos_update)}")
        print("AI_clipper: ", video)
        #Transcribing:
        if file_exists(f"system/{settings['system_variables']['transcribing_name']}"):
            print("Already transcribed...")
            transcribed_text = load(f"system/{settings['system_variables']['transcribing_name']}")
            
        else:
            print("Transcribing...")
            transcribed_text = transcribe_video(video, transcribing_model)
            wright(f"system/{settings['system_variables']['transcribing_name']}", transcribed_text)
        #Chunking
        print("Chunking...")
        chunked_transcribed_text = chunking(transcribed_text, max_token)
        print(f"Chunks created: {len(chunked_transcribed_text)}")

        #AI choosen clips:
        if file_exists(f"system/{settings['system_variables']['AI_name']}"):
            print("Already done AI scanning...")
            AI_output = load(f"system/{settings['system_variables']['AI_name']}")
        else:
            print("Scanning with AI...")
            AI_output = []
            for chunked in chunked_transcribed_text:
                #If the ai cant handle the amount of tokens, we exit the program with a message
                try:
                    output = ai_clipping(chunked, user_query, base_url, model, chunked_transcribed_text, system_message_not_chunked, system_message_chunked)
                except Exception as e:
                    print("AI model failed to process the chunked transcribed text. This may be due to exceeding token limits.")
                    print("Error details:", e)
                    run = False
                    break
                # Append a fresh copy to avoid any unintended shared mutations
                AI_output.append(output)
            wright(f"system/{settings['system_variables']['AI_name']}", AI_output)
        
        #Segment Cleanup
        print("Finding AI scanning in transcribed text...")
        list_of_clips = merge_segments(AI_output, merge_distance)
        print(f"Found: {len(list_of_clips)} Clips!")

        #Video clipping
        if file_exists(f"system/{settings['system_variables']['clips_name']}"):
            print(f"Continuing Clipping...")
            list_of_clips = load(f"system/{settings['system_variables']['clips_name']}")

        else:
            print("Starting Clipping...")
            
        for clip in list_of_clips[:]:
            print(f"Clips left: {len(list_of_clips)}")
            filename = extract_clip(clip, video, clips_output, clips_input, len(list_of_clips))
            list_of_clips.remove(clip)
            wright(f"system/{settings['system_variables']['clips_name']}", list_of_clips)

        #System updating
        iterated += 1
        print(f"System is {iterated/len(videos)*100:.0f}% finished!")
        if os.path.exists(f"system/{settings['system_variables']['AI_name']}"):
            os.remove(f"system/{settings['system_variables']['AI_name']}")
        if os.path.exists(f"system/{settings['system_variables']['transcribing_name']}"):
            os.remove(f"system/{settings['system_variables']['transcribing_name']}")
        if os.path.exists(f"system/{settings['system_variables']['clips_name']}"):
            os.remove(f"system/{settings['system_variables']['clips_name']}")
        if os.path.exists(video):
            os.remove(video)
            
    #Shut down after all videos have been iterated
    print("System finished!")
    run = False