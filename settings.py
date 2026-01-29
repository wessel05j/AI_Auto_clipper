#Python imports
import os

#Component imports
from programs.components.file_exists import file_exists
from programs.components.return_tokens import return_tokens
from programs.components.write import write
from programs.components.load import load

def main():
    #Cloned main variables
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SYSTEM_DIR = os.path.join(BASE_DIR, "system")
    if not os.path.exists(SYSTEM_DIR):
        os.makedirs(SYSTEM_DIR)
    SETTINGS_FILE = os.path.join(SYSTEM_DIR, "settings.json")
    settings_data_contract = {
            "max_tokens": 0,
            "max_ai_tokens": 0,
            "ai_model": "",
            "transcribing_model": "",
            "user_query": "",
            "system_query": "You are an expert transcript editor. Your goal is to extract ALL high-quality clips that match the user query.\n\nInput:\n- JSON transcript: [[start, end, 'text'], ...]\n\nOutput (strict):\n- ONLY JSON: [[start1, end1, score1], [start2, end2, score2], ...]\n- The third element is an integer confidence score (0-10).\n- No prose, no markdown code blocks, no explanations.\n\nRules:\n1. QUANTITY: Find as many relevant clips as possible (up to 5). Do not stop after the first match.\n2. CONTENT FILTERING: REJECT segments containing ONLY non-verbal sounds (grunts, breathing noises, \"oh\", \"ah\", \"mph\", \"ugh\", music, background noise). ONLY include segments with actual spoken sentences or meaningful phrases that form complete thoughts.\n3. BOUNDARIES: Each clip MUST be a complete thought with a clear ending. End clips at natural sentence boundaries (periods, question marks, exclamation points) followed by a pause. NEVER end mid-sentence or when the speaker is transitioning to a new thought. If you see incomplete context at chunk boundaries, be conservative - end earlier at a complete boundary rather than risk cutting off mid-thought.\n4. PRECISION: Use exact timestamps from the input; do not round or estimate.\n5. SCORING: 10 = perfect match with complete sentences, 5 = good clip with clear boundaries, 1 = relevant but marginal. \n6. FALLBACK: Return an empty list [] if no segments contain actual meaningful speech that matches the query.",
            "youtube_list": [],
            "merge_distance": 0,
            "ai_loops": 0,
            "ollama_url": "http://localhost:11434/",
            "temperature": 0.7,
            "channels": [],
            "channels_hours_limit": 0,
            "rerun_temp_files": True
    }

    #Variables
    transcribing_models = ["tiny", "base", "small", "medium", "large"]

    def menu(max_tokens, max_ai_tokens, ai_model, transcribing_model, user_query, system_query, youtube_list, merge_distance, ai_loops, ollama_url, temperature, channels, channels_hours_limit, rerun_temp_files):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("<-----MENU----->")
        print(f"<-----1. Max Tokens: {max_tokens} (rest for model(cannot be changed): {max_ai_tokens})")
        print(f"<-----2. AI Model: {ai_model}")
        print(f"<-----3. Transcribing Model: {transcribing_model}")
        print(f"<-----4. User Query: {user_query}")
        print(f"<-----5. System Query: {system_query[:30]}...")
        print(f"<-----6. YouTube Links: {youtube_list}")
        print(f"<-----7. Merge Distance: {merge_distance}")
        print(f"<-----8. AI Loops: {ai_loops}")
        print(f"<-----9. Ollama_url: {ollama_url}")
        print(f"<-----10. Temperature: {temperature}")
        print(f"<-----11. Channels to monitor: {channels}")
        print(f"<-----12. Channel Hours Limit: {channels_hours_limit}")
        print(f"<-----13. Rerun Temp Files: {rerun_temp_files}")
        print("<-----0. Exit settings\n")

    #Setup needed
    if not file_exists(SETTINGS_FILE):
        os.makedirs(SYSTEM_DIR, exist_ok=True)
        try:
            template_settings = settings_data_contract.copy()
            print("Welcome to the AI Auto Clipper setup!")

            ai_model = input("Please enter the EXACT Ollama model name (e.g., 'gpt-oss:20b'): ")
            transcribing_model = input("Please enter the transcribing model name (e.g.,'tiny', 'base', 'small', 'medium', 'large'): ").lower()
            while transcribing_model not in transcribing_models:
                transcribing_model = input(f"Invalid model name. Please choose from {transcribing_models}: ").lower()
            user_query = input("Please enter your query for the AI to choose clips (e.g., 'Find all clips where someone is talking about cats'): ")
            while user_query.strip() == "":
                user_query = input("User query cannot be empty. Please enter your query: ")
            max_tokens = 0
            while max_tokens <= 0:
                try:
                    max_tokens = int(input("Please enter the maximum tokens your AI model can handle (total model limit): "))
                    while max_tokens <= 0:
                        max_tokens = int(input("Max tokens must be a positive integer. Please enter again: "))
                    max_ai_tokens = max_tokens*0.4
                    max_tokens = (max_tokens*0.6 - (return_tokens(template_settings["system_query"]) + return_tokens(user_query)))
                except Exception as e:
                    print(f"Invalid input for max tokens. Error: {e}")
            merge_distance = 0
            while merge_distance <= 0:
                try:
                    print("Merge distance works as follows: \nIf two clips are within the merge distance (in seconds), they will be merged into one clip.\nA higher merge distance can lead to longer videos with more context, while a lower merge distance results in shorter, more concise clips.")
                    merge_distance = int(input("Please enter the merge distance in seconds): "))
                except Exception as e:
                    print(f"Invalid input for merge distance. Error: {e}")
            ai_loops = 0
            while ai_loops <= 0:
                try:
                    print("AI loops determine how many times the AI will scan through the transcript to find relevant clips.\nMore loops can lead to better clip selection but will increase processing time.")
                    ai_loops = int(input("Please enter how many times the AI should scan through the transcript for clips: "))
                except Exception as e:
                    print(f"Invalid input for AI loops. Error: {e}")
            temperature = -1.0
            while temperature < 0.0 or temperature > 1.0:
                try:
                    temperature = float(input("Please enter the temperature for the AI model (0.0 to 1.0, where 0.0 is deterministic and 1.0 is creative): "))
                except Exception as e:
                    print(f"Invalid input for temperature. Error: {e}")
            
            #Saving settings
            template_settings["max_tokens"] = max_tokens 
            template_settings["max_ai_tokens"] = max_ai_tokens
            template_settings["ai_model"] = ai_model 
            template_settings["transcribing_model"] = transcribing_model 
            template_settings["user_query"] = user_query 
            template_settings["merge_distance"] = merge_distance
            template_settings["ai_loops"] = ai_loops
            template_settings["temperature"] = temperature
            write(SETTINGS_FILE, template_settings)
        except Exception as e:
            print(f"Error during setup: {e}")
            return

    while True:
        try:
            current_settings = load(SETTINGS_FILE)
            max_tokens = current_settings["max_tokens"]
            ai_model = current_settings["ai_model"]
            max_ai_tokens = current_settings["max_ai_tokens"]
            transcribing_model = current_settings["transcribing_model"]
            user_query = current_settings["user_query"]
            system_query = current_settings["system_query"]
            youtube_list = current_settings["youtube_list"]
            merge_distance = current_settings["merge_distance"]
            ai_loops = current_settings["ai_loops"]
            ollama_url = current_settings["ollama_url"]
            temperature = current_settings["temperature"]
            channels = current_settings["channels"]
            channels_hours_limit = current_settings["channels_hours_limit"]
            rerun_temp_files = current_settings["rerun_temp_files"]
            menu(max_tokens, max_ai_tokens, ai_model, transcribing_model, user_query, system_query, youtube_list, merge_distance, ai_loops, ollama_url, temperature, channels, channels_hours_limit, rerun_temp_files)
            new_settings = load(SETTINGS_FILE)

            choice = input("Input: ").strip()
            if choice == "1":
                try:
                    raw_max = int(input("Enter new Max Tokens (total model limit): ").strip())*0.6
                    if raw_max <= 0:
                        print("Max tokens must be a positive integer.")
                        continue
                    elif raw_max == False:
                        continue
                    max_ai_tokens = raw_max*0.4
                    new_settings["max_ai_tokens"] = max_ai_tokens
                    overhead = (
                    return_tokens(current_settings["system_query"]) +
                    return_tokens(current_settings["user_query"])
                    )
                    max_tokens = (raw_max - overhead)
                    new_settings["max_tokens"] = max_tokens 
                except Exception as e:
                    print(f"Make sure its an integer: {e}")
                    continue 
            elif choice == "2":
                ai_model = input("Enter new AI Model (Enter to skip): ")
                if ai_model.strip() == "":
                    continue
                new_settings["ai_model"] = ai_model
            elif choice == "3":
                transcribing_model = input("Please enter the transcribing model name (e.g.,'tiny', 'base', 'small', 'medium', 'large'): ").lower()
                if transcribing_model in transcribing_models:
                    new_settings["transcribing_model"] = transcribing_model
            elif choice == "4":
                user_query = input("Enter new User Query (Enter to skip): ")
                if user_query.strip() == "":
                    continue
                new_settings["user_query"] = user_query
            elif choice == "5":
                print("NB! System query is not recommended to be changed unless you know what you are doing.")
                #Printing the current system query for reference
                print(f"Current System Query:\n{system_query}\n")
                system_query = input("Enter new System Query (Enter to skip): ")
                if system_query.strip() == "":
                    continue
                new_settings["system_query"] = system_query
            elif choice == "6":
                question = input("Would you like to add or create a new list (add/new/enter to skip)? ").lower()
                if question == "":
                    continue
                elif question in ["a", "add"]:
                    try:
                        youtube_list_input = input("Enter new YouTube Links separated by commas: ").strip()
                        current_youtube_list = current_settings["youtube_list"]
                        youtube_list = list(current_youtube_list)  # Copy the current list
                        new_list = [link.strip() for link in youtube_list_input.split(",")]
                        youtube_list.extend(new_list)
                        new_settings["youtube_list"] = youtube_list
                    except Exception as e:
                        print(f"Error adding links: {e}")
                        continue
                elif question in ["n", "new"]:
                    try:
                        youtube_list_input = input("Enter new YouTube Links separated by commas: ").strip()
                        youtube_list = [link.strip() for link in youtube_list_input.split(",")]
                        new_settings["youtube_list"] = youtube_list
                    except Exception as e:
                        print(f"Error creating new list: {e}")
                        continue
                else:
                    print("Please write on of the following n, new, a or add")
            elif choice == "7":
                try:
                    merge_distance = int(input("Enter new Merge Distance (highly impacts length of video, enter to skip): ").strip())
                    if merge_distance == False:
                        continue
                    new_settings["merge_distance"] = merge_distance
                except Exception as e:
                    print(f"Make sure its an integer: {e}")
                    continue
            elif choice == "8":
                try:
                    ai_loops = int(input("Enter new AI Loops (how many times the AI should scan the transcript, enter to skip): ").strip())
                    if ai_loops <= 0:
                        print("ai_loops cannot be 0 or lower")
                        continue
                    elif ai_loops == False:
                        continue
                    new_settings["ai_loops"] = ai_loops
                except Exception as e:
                    print(f"Make sure its an integer: {e}")
                    continue
            elif choice == "9":
                print("Ollama URL is the local or remote address where your Ollama server is hosted.")
                print("Be precise with the format, including http:// or https:// and the correct port (default is usually 11434).")
                ollama_url = input("Enter new Ollama URL (e.g., http://localhost:11434/), enter to skip: ").strip()
                if ollama_url == "":
                    continue
                new_settings["ollama_url"] = ollama_url
            elif choice == "10":
                if temperature == "":
                    continue
                else:
                    try:
                        temperature = float(input("Please enter the temperature for the AI model (0.0 to 1.0, where 0.0 is deterministic and 1.0 is creative, enter to skip): "))
                        if temperature < 0.0 or temperature > 1.0:
                            print("Temperature must be between 0.0 and 1.0.")
                        if temperature == False:
                            continue
                    except Exception as e:
                        print(f"Invalid input for temperature. Error: {e}")
                new_settings["temperature"] = temperature
            elif choice == "11":
                print("Enter the YouTube channel URLs you want to monitor for recent videos using fetch_yt_links.py.")
                print("Channel URL must have /videos at the end... e.g., https://www.youtube.com/@CheekyCrypto/videos")
                channels_input = input("Enter YouTube channel URLs to monitor, separated by commas, enter to skip: ").strip()
                if channels_input == "":
                    continue
                channels = [link.strip() for link in channels_input.split(",") if link.strip() != ""]
                for channel in channels:
                    if "/videos" not in channel:
                        print(f"Channel URL '{channel}' is invalid. It must end with '/videos'. Please try again.")
                        continue
                new_settings["channels"] = channels
            elif choice == "12":
                try:
                    channels_hours_limit = int(input("Enter the hours limit to fetch recent videos from channels, enter to skip: ").strip())
                    if channels_hours_limit == False:
                        continue
                    new_settings["channels_hours_limit"] = channels_hours_limit
                except Exception as e:
                    print(f"Make sure its an integer: {e}")
                    continue
            elif choice == "13":
                print("Rerun Temp Files determines whether the program should reprocess temporary files created during previous runs.")
                rerun_input = input("Would you like to enable rerunning temp files? (yes/no, enter to skip): ").strip().lower()
                if rerun_input == "":
                    continue
                elif rerun_input in ["yes", "y"]:
                    new_settings["rerun_temp_files"] = True
                elif rerun_input in ["no", "n"]:
                    new_settings["rerun_temp_files"] = False
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")
                    continue
            elif choice == "0":
                break
            else:
                print("Invalid choice. Please try again.")
                continue

            #Save settings
            write(SETTINGS_FILE, new_settings)
        except Exception as e:
            print(f"Error during settings: {e}")
            return

if __name__ == "__main__":
    main()
