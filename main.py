import os
import json

from programs.transcribing import transcribe_video
from programs.ai_scanning import ai_clipping
from programs.chunking import chunking
from programs.merge_segments import merge_segments
from programs.extract_clip import extract_clip
from programs.scan_videos import scan_videos
from programs.variable_checker import variable_checker
from programs.yt_downloader import yt_downloader

#Variables needed:
user_query = '''
    Identify and extract long-form motivational clips (60 seconds or longer). 
    Only select segments where the speaker is directly addressing the audience in an uplifting, encouraging, or inspiring way. 
    You may also include interesting or insightful moments, but motivation and audience-directed uplift must be the priority.
'''

base_url = "http://127.0.0.1:1234/v1"
model = "openai/gpt-oss-20b"
clips_input = "videos/"
clips_output = "output/"
transcribing_model = "small" #tiny, base, small, medium, large, large-v3, large-v3-turbo
max_token = 10000 #Max tokens per chunk for AI processing


#optional youtube downloading:
youtube_list = []


#Checking variables to be correct:
checked = variable_checker(user_query, base_url, model, clips_input, clips_output, transcribing_model, max_token)
if len(checked) > 0:
    print(f"Some variables were not declared. Please fix: {checked}")
    run = False
else:
    run = True

while run:
    #Donwload youtube videos if the user have declared them:
    videos = scan_videos(clips_input)
    if len(youtube_list) > 0:
        for link in youtube_list:
            yt_downloader(link, clips_input) #Downloading the videos to input folder
    
    #Collection all videos for clipping:
    print("Collection videos...")
    iterated = 0
    print("System is 0% finished")
    for video in videos: 
        videos_update = scan_videos(clips_input)
        print(f"Videos left: {videos_update}")
        print("AI_clipper: ", video)
        #Transcribing:
        if os.path.exists("system/transcribed.json") and os.path.getsize("system/transcribed.json") > 0:
            print("Already transcribed...")
            with open("system/transcribed.json", "r") as f:
                transcribed_text = json.load(f)
        else:
            print("Transcribing...")
            transcribed_text = transcribe_video(video, transcribing_model)
            with open("system/transcribed.json", "w") as f:
                json.dump(transcribed_text, f)

        #Chunking
        print("Chunking...")
        chunked_transcribed_text = chunking(transcribed_text, max_token)
        print(f"Chunks created: {len(chunked_transcribed_text)}")

        #AI choosen clips:
        if os.path.exists("system/AI.json") and os.path.getsize("system/AI.json") > 0:
            print("Already done AI scanning...")
            with open("system/AI.json", "r") as f:
                AI_output = json.load(f)
        else:
            print("Scanning with AI...")
            AI_output = []
            for chunked in chunked_transcribed_text:
                output = ai_clipping(chunked, user_query, base_url, model, chunked_transcribed_text)
                # Append a fresh copy to avoid any unintended shared mutations
                AI_output.append(output)
            with open("system/AI.json", "w") as f:
                json.dump(AI_output, f)

        #Segment Cleanup
        print("Finding AI scanning in transcribed text...")
        list_of_clips = merge_segments(AI_output, 30)
        print(f"Found: {len(list_of_clips)} Clips!")

        #Video clipping
        print(f"Clips left: {len(list_of_clips)}")
        if os.path.exists("system/Clips.json") and os.path.getsize("system/Clips.json") > 0:
            print(f"Continuing Clipping...")
            with open("system/Clips.json", "r") as f:
                list_of_clips = json.load(f)

        else:
            print("Starting Clipping...")
            
        for clip in list_of_clips[:]:
            print(f"Clips left: {len(list_of_clips)}")
            extract_clip(clip, video, clips_output, clips_input, len(list_of_clips))
            list_of_clips.remove(clip)
            with open("system/Clips.json", "w") as f:
                json.dump(list_of_clips, f)
                

        #System updating
        iterated += 1
        print(f"System is {iterated/len(videos)*100:.0f}% finished!")
        if os.path.exists("system/AI.json"):
            os.remove("system/AI.json")
        if os.path.exists("system/transcribed.json"):
            os.remove("system/transcribed.json")
        if os.path.exists("system/Clips.json"):
            os.remove("system/Clips.json")
        if os.path.exists(video):
            os.remove(video)
    
    #Shut down after all videos have been iterated
    print("System finished!")
    run = False