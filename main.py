import os
import json

#Core files
from programs.variable_checker import variable_checker
from programs.scan_videos import scan_videos
from programs.yt_downloader import yt_downloader
from programs.transcribing import transcribe_video
from programs.chunking import chunking
from programs.ai_scanning import ai_clipping
from programs.merge_segments import merge_segments
from programs.extract_clip import extract_clip

#UI cleanup files
from programs.interact_w_json import interact_w_json
from programs.file_exists import file_exists


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
accuracy_testing = False #If true, will re-transcribe the clip at the end to check for accuracy
accuracy_model = "medium" #Preferrably large, takes time but the bigger the better chance for accuracy in the clips removing any clutter outside choosen clip... 
max_token = 10000 #Max tokens per chunk for AI processing
youtube_list = [] #optional youtube downloading:



#Checking variables to be correct (To be updated):
checked = variable_checker(user_query, base_url, model, clips_input, clips_output, transcribing_model, accuracy_model, max_token, youtube_list)
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
    print(f"System is 0% finished")
    for video in videos: 
        videos_update = scan_videos(clips_input)
        print(f"Videos left: {videos_update}")
        print("AI_clipper: ", video)
        #Transcribing:
        if file_exists("system/transcribed.json"):
            print("Already transcribed...")
            transcribed_text = interact_w_json("system/transcribed.json", "r", None)
            
        else:
            print("Transcribing...")
            transcribed_text = transcribe_video(video, transcribing_model)
            interact_w_json("system/transcribed.json", "w", transcribed_text)

        #Chunking
        print("Chunking...")
        chunked_transcribed_text = chunking(transcribed_text, max_token)
        print(f"Chunks created: {len(chunked_transcribed_text)}")

        #AI choosen clips:
        if file_exists("system/AI.json"):
            print("Already done AI scanning...")
            AI_output = interact_w_json("system/AI.json", "r", None)
        else:
            print("Scanning with AI...")
            AI_output = []
            for chunked in chunked_transcribed_text:
                output = ai_clipping(chunked, user_query, base_url, model, chunked_transcribed_text)
                # Append a fresh copy to avoid any unintended shared mutations
                AI_output.append(output)
            interact_w_json("system/AI.json", "w", AI_output)

        #Segment Cleanup
        print("Finding AI scanning in transcribed text...")
        list_of_clips = merge_segments(AI_output, 30)
        print(f"Found: {len(list_of_clips)} Clips!")

        #Video clipping
        print(f"Clips left: {len(list_of_clips)}")
        if file_exists("system/Clips.json"):
            print(f"Continuing Clipping...")
            list_of_clips = interact_w_json("system/Clips.json", "r", None)

        else:
            print("Starting Clipping...")
            
        for clip in list_of_clips[:]:
            print(f"Clips left: {len(list_of_clips)}")
            filename = extract_clip(clip, video, clips_output, clips_input, len(list_of_clips))

            #Testing Accuracy:
            if accuracy_testing == False:
                print("Skipping clip accuracy testing...")
                list_of_clips.remove(clip)
                interact_w_json("system/Clips.json", "w", list_of_clips)
            else:
                print("Testing clip accuracy...")
                if file_exists("system/accuracy_transcribed.json") == False:
                    transcribed_text = transcribe_video(filename, accuracy_model)
                    interact_w_json("system/accuracy_transcribed.json", "w", transcribed_text)
                else:
                    transcribed_text = interact_w_json("system/accuracy_transcribed.json", "r", None)

                last_segment = transcribed_text[-1]
                example = "Yeah Because. "
                if len(last_segment[2]) == len(example) or len(last_segment[2]) < len(example):
                    print("Clip inaccurate, removing last segment and re-making clip...")
                    #Remove the last segment and make the clip again
                    transcribed_text.remove(last_segment)

                    #Removing the current clip
                    os.remove(filename)
                    new_end = last_segment[0]
                    new_clip = [clip[0], new_end]
                    filtered_clip = extract_clip(new_clip, video, clips_output, clips_input, len(list_of_clips))
                    list_of_clips.remove(new_clip)
                    interact_w_json("system/Clips.json", "w", list_of_clips)
            
        


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