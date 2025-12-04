def setup_stage():
    #Python imports
    import os
    

    #Component imports
    from programs.components.file_exists import file_exists
    from programs.components.return_tokens import return_tokens
    from programs.components.wright import wright
    from programs.components.load import load
    from programs.setup_stage.interact_w_ai import interact_w_ai
    from programs.setup_stage.max_tokens_ai_check import max_tokens_ai_check

    #Variables
    setup_needed = False
    main_run = False
    settings_path = "system/settings.json"

    #Ensure we have a system folder
    if not os.path.exists("system/"):
        os.makedirs("system/")

    #Load settings
    if not file_exists(settings_path):
        template_settings = {
            "setup_variables": {
                "max_tokens": 0,
                "output_folder": "",
                "input_folder": "",
                "ai_model": "",
                "base_url": "",
                "transcribing_model": "",
                "user_query": "",
                "youtube_list": [],
                "merge_distance": 30
            },
            "system_variables": {
                "version": "1.0.0",
                "transcribing_name": "transcribed.json",
                "AI_name": "AI.json",
                "clips_name": "clips.json",
                "AI_instructions_w_chunking": '''
                    You are a Context-Aware Transcript Editor. Your purpose is to extract semantically complete conversational segments based on a user query.

                    Transcript Format:
                    You receive a transcript as JSON: [[start, end, "text"], ...] where start and end are seconds (floats) and text is the spoken content.

                    Strict Output Format:
                    Return ONLY a JSON array of [start, end] pairs: [[start1, end1], [start2, end2], ...].
                    No reasoning, no explanations, no extra keys, no comments.
                    The JSON must be syntactically valid and parseable.

                    Time Precision:
                    - Preserve the full floating-point precision of the timestamps.
                    - Do NOT round start or end times to a fixed number of decimals.
                    - If the transcript shows times like 10.3725, use that full precision in your output.

                    Thought Unit and Completeness Rules:
                    - Each clip must be a semantically complete thought with a beginning (setup), middle (discussion), and end (resolution or clear pause).
                    - Never start mid-sentence or mid-word; prefer natural sentence or paragraph beginnings.
                    - Do not end mid-sentence, mid-list, or in the middle of an explanation; stop at a natural pause or conclusion.

                    Length and Relevance:
                    - Prefer longer, context-rich clips that remain clearly relevant to the user query.
                    - Typical length: 30–360 seconds when possible.
                    - Short clips (<30–60 seconds) are allowed only when no longer coherent segment meaningfully fits the query.
                    
                    Chunk Awareness:
                    - You may see only a portion of the full transcript at a time.
                    - Avoid starting or ending clips around the very first segments of the visible transcript if it is clearly mid-thought.
                    - NEVER make a clip with the last segment of an transcript.
                    - If a topic obviously continues beyond the visible end, end your clip at the last natural boundary you can see, not at a mid-sentence cutoff.

                    Query Conditioning:
                    - Select only clips that are clearly relevant to the user query.
                    - Some extra surrounding context before and after is allowed if it preserves a natural conversational flow.
                    ''',
                "Ai_instructions": '''
                    You are a Context-Aware Transcript Editor. Your purpose is to extract semantically complete conversational segments based on a user query.

                    Transcript Format:
                    You receive a transcript as JSON: [[start, end, "text"], ...] where start and end are seconds (floats) and text is the spoken content.

                    Strict Output Format:
                    Return ONLY a JSON array of [start, end] pairs: [[start1, end1], [start2, end2], ...].
                    No reasoning, no explanations, no extra keys, no comments.
                    The JSON must be syntactically valid and parseable.

                    Time Precision:
                    - Preserve the full floating-point precision of the timestamps.
                    - Do NOT round start or end times to a fixed number of decimals.
                    - If the transcript shows times like 10.3725, use that full precision in your output.

                    Thought Unit and Completeness Rules:
                    - Each clip must be a semantically complete thought with a beginning (setup), middle (discussion), and end (resolution or clear pause).
                    - Never start mid-sentence or mid-word; prefer natural sentence or paragraph beginnings.
                    - Do not end mid-sentence, mid-list, or in the middle of an explanation; stop at a natural pause or conclusion.

                    Length and Relevance:
                    - Prefer longer, context-rich clips that remain clearly relevant to the user query.
                    - Typical length: 30–360 seconds when possible.
                    - Short clips (<30–60 seconds) are allowed only when no longer coherent segment meaningfully fits the query.

                    Query Conditioning:
                    - Select only clips that are clearly relevant to the user query.
                    - Some extra surrounding context before and after is allowed if it preserves a natural conversational flow.
                    '''
            }
        }
        setup_needed = True

    #Setting up settings if needed
    if setup_needed:
        print("Welcome to the AI Auto Clipper setup!")
        output_folder = input("Please enter the output folder path: ")
        if "/" not in output_folder and "\\" not in output_folder:
            output_folder = output_folder + "/"
        if not os.path.exists(output_folder):
            #Create output folder if it doesn't exist
            os.makedirs(output_folder)


        input_folder = input("Please enter the input folder path: ")
        if "/" not in input_folder and "\\" not in input_folder:
            input_folder = input_folder + "/"
        if not os.path.exists(input_folder):
            #Create input folder if it doesn't exist
            os.makedirs(input_folder)
        
        ai_model = input("Please enter the AI model name, copypaste from LM studio (e.g., 'gpt-4o'): ")
        base_url = input("Please enter the base URL for the AI model (LM studio) Make sure to be running the AI: ")
        if "/v1" not in base_url:
            base_url = base_url + "/v1"
        print("Testing AI connection...")
        while True:
            try:
                interaction = interact_w_ai(base_url, ai_model)
                print("AI connection successful! We got response: ", interaction)
                break
            except Exception as e:
                print("AI connection failed. Please check the base URL and model name.")
                print("Error details: ", str(e))

        transcribing_models = ["tiny", "base", "small", "medium", "large"]
        transcribing_model = input("Please enter the transcribing model name (e.g.,'tiny', 'base', 'small', 'medium', 'large'): ")
        while transcribing_model not in transcribing_models:
            transcribing_model = input(f"Invalid model name. Please choose from {transcribing_models}: ")
        
        user_query = input("Please enter your query for the AI to choose clips (e.g., 'Find all clips where someone is talking about cats'): ")

        print("Finding out how many tokens the AI can handle (This can take some while)...")
        while True:
            try:
                max_tokens = int(max_tokens_ai_check(base_url, ai_model)) - return_tokens(template_settings["system_variables"]["AI_instructions_w_chunking"]) - return_tokens(user_query)
                print(f"AI can handle up to {max_tokens} tokens per prompt and response.")
                break
            except Exception as e:
                print("Failed to determine max tokens from AI, Trying again... Ai model might be weak at this.")
                print("Error details: ", str(e))
                max_tokens = int(input("Please input what max tokens is: "))
        
        #Now creating your settings folder and file
        template_settings["setup_variables"]["max_tokens"] = max_tokens #int
        template_settings["setup_variables"]["output_folder"] = output_folder #str
        template_settings["setup_variables"]["input_folder"] = input_folder #str
        template_settings["setup_variables"]["ai_model"] = ai_model #str
        template_settings["setup_variables"]["base_url"] = base_url #str
        template_settings["setup_variables"]["transcribing_model"] = transcribing_model #str
        template_settings["setup_variables"]["user_query"] = user_query #str
        template_settings["setup_variables"]["youtube_list"] = [] #list
        wright(settings_path, template_settings)
    
    settings = load(settings_path)
    skip = input("Are you currently running a session and just want to skip booting stage? (y/N): ").strip().lower()
    if skip in ["n", "no"]:

        #Checking the variables
        print("Checking your settings for potential issues...")
        value_errors = []
        #Issue here: check with settings["setup_variables"]["max_tokens"]
        if settings["setup_variables"]["max_tokens"] <= 500:
            value_errors.append("Your AI max tokens setting is too low. Please choose higher capacity AI.")
        
        if not os.path.exists(settings["setup_variables"]["output_folder"]):
            value_errors.append("The specified output folder does not exist: " + settings["setup_variables"]["output_folder"])
        
        if not os.path.exists(settings["setup_variables"]["input_folder"]):
            value_errors.append("The specified input folder does not exist: " + settings["setup_variables"]["input_folder"])

        if settings["setup_variables"]["ai_model"].strip() == "":
            value_errors.append("The AI model is not set.")

        if settings["setup_variables"]["base_url"].strip() == "":
            value_errors.append("The AI base URL is not set.")
        
        if settings["setup_variables"]["transcribing_model"].strip() == "":
            value_errors.append("The transcribing model is not set.")
        
        if settings["setup_variables"]["user_query"].strip() == "":
            value_errors.append("The user query is not set.")
        
        if settings["setup_variables"]["youtube_list"] is None:
            value_errors.append("The youtube list is not set.")
        
        print(f"Found {len(value_errors)} potential issues.\n")
        if len(value_errors) > 0:
            for error in value_errors:
                print("Potential issue: ", error)

        #Boot menu
        max_tokens = settings["setup_variables"]["max_tokens"]
        output_folder = settings["setup_variables"]["output_folder"]
        input_folder = settings["setup_variables"]["input_folder"]
        ai_model = settings["setup_variables"]["ai_model"]
        base_url = settings["setup_variables"]["base_url"]
        transcribing_model = settings["setup_variables"]["transcribing_model"]
        user_query = settings["setup_variables"]["user_query"]
        youtube_list = settings["setup_variables"]["youtube_list"]
        merge_distance = settings["setup_variables"]["merge_distance"]

        answer = input("Would you like to edit some settings before proceeding? (Y/n): ").strip().lower()
        if answer in ["y", "yes"]:
            while True:
                print("Which setting would you like to edit?")
                print(f"1. Max Tokens: {max_tokens}")
                print(f"2. Output Folder: {output_folder}")
                print(f"3. Input Folder: {input_folder}")
                print(f"4. AI Model: {ai_model}")
                print(f"5. Base URL: {base_url}")
                print(f"6. Transcribing Model: {transcribing_model}")
                print(f"7. User Query: {user_query}")
                print(f"8. YouTube Links: {youtube_list}")
                print(f"9. Merge Distance (highly impacts length of video): {merge_distance}")
                print("0. Boot up with current settings")

                choice = input("Enter the number of your choice: ").strip()
                if choice == "1":
                    manual = input("Do you want to manually enter max tokens (Y/n): ").strip().lower()
                    if manual in ["y", "yes"]:
                        try:
                            max_tokens = int(input("Enter new Max Tokens: ").strip()) - return_tokens(settings["system_variables"]["AI_instructions_w_chunking"]) - return_tokens(settings["system_variables"]["user_query"])
                        except Exception as e:
                            print(f"Make sure its an integer: {e}")
                    else:
                        try:
                            max_tokens = max_tokens_ai_check(base_url, ai_model) - return_tokens(settings["system_variables"]["AI_instructions_w_chunking"]) - return_tokens(user_query)
                            print(f"AI can handle up to {max_tokens} tokens per prompt and response.")
                        except Exception as e:
                            print(f"AI at its task: {e}")
                            
                elif choice == "2":
                    output_folder = input("Please enter the output folder path: ")
                    if "/" not in output_folder and "\\" not in output_folder:
                        output_folder = output_folder + "/"
                    if not os.path.exists(output_folder):
                        while not os.path.exists(output_folder):
                            print("The specified output folder does not exist. Please try again.")
                            output_folder = input("Please enter the output folder path: ")
                elif choice == "3":
                    input_folder = input("Please enter the input folder path: ")
                    if "/" not in input_folder and "\\" not in input_folder:
                        input_folder = input_folder + "/"
                    if not os.path.exists(input_folder):
                        while not os.path.exists(input_folder):
                            print("The specified input folder does not exist. Please try again.")
                            input_folder = input("Please enter the input folder path: ")
                elif choice == "4":
                    ai_model = input("Enter new AI Model: ")
                elif choice == "5":
                    base_url = input("Enter new Base URL: ")
                    if "/v1" not in base_url:
                        base_url = base_url + "/v1"
                elif choice == "6":
                    transcribing_model = input("Enter new Transcribing Model: ")
                elif choice == "7":
                    user_query = input("Enter new User Query: ")
                elif choice == "8":
                    question = input("Would you like to add or create a new list (add/new)?").lower()
                    if question in ["a", "add"]:
                        youtube_list_input = input("Enter new YouTube Links separated by commas: ").strip()
                        current_youtube_list = settings["setup_variables"]["youtube_list"]
                        for link in current_youtube_list:
                            youtube_list.append(link)
                        new_list = [link.strip() for link in youtube_list_input.split(",")]
                        for new_link in new_list:
                            youtube_list.append(new_link)
                    elif question in ["n", "new"]:
                        youtube_list_input = input("Enter new YouTube Links separated by commas: ").strip()
                        youtube_list = [link.strip() for link in youtube_list_input.split(",")]
                    else:
                        print("Please write on of the following n, new, a or add")
                elif choice == "9":
                    try:
                        merge_distance = int(input("Enter new Merge Distance (highly impacts length of video): ").strip())
                    except Exception as e:
                        print(f"Make sure its an integer: {e}")
                elif choice == "0":
                    break
                else:
                    print("Invalid choice. Please try again.")
                #Save updated settings
                new_settings = load(settings_path)
                new_settings["setup_variables"]["max_tokens"] = max_tokens
                new_settings["setup_variables"]["output_folder"] = output_folder
                new_settings["setup_variables"]["input_folder"] = input_folder
                new_settings["setup_variables"]["ai_model"] = ai_model
                new_settings["setup_variables"]["base_url"] = base_url
                new_settings["setup_variables"]["transcribing_model"] = transcribing_model
                new_settings["setup_variables"]["user_query"] = user_query
                new_settings["setup_variables"]["youtube_list"] = youtube_list
                new_settings["setup_variables"]["merge_distance"] = merge_distance
                wright(settings_path, new_settings)
                

    print("Booting up...")
    settings = load(settings_path)
    base_url = settings["setup_variables"]["base_url"]
    ai_model = settings["setup_variables"]["ai_model"]
    try:
        interact_w_ai(base_url, ai_model)
        main_run = True
    except Exception as e:
        print(f"Error interacting with AI (Make sure the AI service is available): {e}")
    
    return main_run
            
        
    
    

    