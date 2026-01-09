def setup_stage(SETTINGS_FILE: str):
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

    def menu(max_tokens, ai_model, transcribing_model, user_query, youtube_list, merge_distance):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("<-----MENU----->")
        print(f"<-----1. Max Tokens: {max_tokens}")
        print(f"<-----2. AI Model: {ai_model}")
        print(f"<-----3. Transcribing Model: {transcribing_model}")
        print(f"<-----4. User Query: {user_query}")
        print(f"<-----5. YouTube Links: {youtube_list}")
        print(f"<-----6. Merge Distance (highly impacts length of video): {merge_distance}")
        print("<-----0. Boot up with current settings")
        print("<-----MENU----->")

    def issue_checker(SETTINGS_FILE: str):
        value_errors = []
        #Issue here: check with settings["setup_variables"]["max_tokens"]
        if SETTINGS_FILE["setup_variables"]["max_tokens"] <= 500:
            value_errors.append("There might be an issue with max tokens being too low or nothing at all.")

        if SETTINGS_FILE["setup_variables"]["ai_model"].strip() == "":
            value_errors.append("The AI model is not set.")

        try:
            interact_w_ai(SETTINGS_FILE["setup_variables"]["ai_model"])
        except Exception as e:
            value_errors.append(f"Error interacting with AI: {e}")
        
        if SETTINGS_FILE["setup_variables"]["transcribing_model"].strip() == "":
            value_errors.append("The transcribing model is not set.")
        
        if SETTINGS_FILE["setup_variables"]["user_query"].strip() == "":
            value_errors.append("The user query is not set.")
        
        if SETTINGS_FILE["setup_variables"]["youtube_list"] is None:
            value_errors.append("The youtube list is not set.")
        
        if len(value_errors) > 0:
            for error in value_errors:
                print("Potential issue: ", error)

    #Load settings
    if not file_exists(SETTINGS_FILE):
        template_settings = {
            "setup_variables": {
                "max_tokens": 0,
                "ai_model": "",
                "transcribing_model": "",
                "user_query": "",
                "youtube_list": [],
                "merge_distance": 30
            },
            "system_variables": {
                "AI_instruction": '''
                    You are a transcript clip selector.

                    Input:
                    - JSON transcript: [[start, end, "text"], ...] with start/end in seconds.

                    Output (strict):
                    - ONLY JSON: [[start1, end1, score1], [start2, end2, score2], ...]
                    - The third element in each list is a confidence score (0-10) representing how well the clip matches the user query.
                    - No explanations, no extra keys, no comments.

                    Rules:
                    - Keep full timestamp precision; do not round.
                    - Each clip must be a complete thought: clear beginning, middle, and end.
                    - The confidence score must be an integer from 0 (poor match) to 10 (perfect match).
                    - Never start or end in the middle of a word or sentence.
                    - Prefer natural pauses and paragraph boundaries for cut points.
                    - Prefer context-rich clips (roughly 30–360 seconds) that clearly match the user query.
                    - If no segments match the query, return an empty list: [].
                    '''
            }
        }
        setup_needed = True

    #Setting up settings if needed
    if setup_needed:
        print("Welcome to the AI Auto Clipper setup!")

        ai_model = input("Please enter the Ollama model name (e.g., 'gpt-oss:20b'): ")
        print("Testing AI connection...")
        while True:
            try:
                interact_w_ai(ai_model)
                print("AI connection successful!")
                break
            except Exception as e:
                print("AI connection failed. Please check the model name and ensure Ollama is running.")
                print("Error details: ", str(e))
            #Waiting a few seconds before retrying
            print(input("Press Enter to retry..."))

        transcribing_models = ["tiny", "base", "small", "medium", "large"]
        transcribing_model = input("Please enter the transcribing model name (e.g.,'tiny', 'base', 'small', 'medium', 'large'): ").lower()
        while transcribing_model not in transcribing_models:
            transcribing_model = input(f"Invalid model name. Please choose from {transcribing_models}: ").lower()
        
        user_query = input("Please enter your query for the AI to choose clips (e.g., 'Find all clips where someone is talking about cats'): ")
        print("Finding out how many tokens the AI can handle (This can take some while)...")
        while True:
            try:
                max_tokens = (int(max_tokens_ai_check(ai_model)) - return_tokens(template_settings["system_variables"]["AI_instruction"]) - return_tokens(user_query))*0.6
                print(f"AI can handle up to {max_tokens} tokens per prompt and response.")
                break
            except Exception as e:
                print("Failed to determine max tokens from AI, Trying again... Ai model might be weak at this.")
                print("Error details: ", str(e))
                max_tokens = (int(input("Please input what max tokens is: ")) - return_tokens(template_settings["system_variables"]["AI_instruction"]) - return_tokens(user_query))*0.6
        
        template_settings["setup_variables"]["max_tokens"] = max_tokens 
        template_settings["setup_variables"]["ai_model"] = ai_model 
        template_settings["setup_variables"]["transcribing_model"] = transcribing_model 
        template_settings["setup_variables"]["user_query"] = user_query 
        template_settings["setup_variables"]["youtube_list"] = [] 
        wright(SETTINGS_FILE, template_settings)

    current_settings = load(SETTINGS_FILE)
    max_tokens = current_settings["setup_variables"]["max_tokens"]
    ai_model = current_settings["setup_variables"]["ai_model"]
    transcribing_model = current_settings["setup_variables"]["transcribing_model"]
    user_query = current_settings["setup_variables"]["user_query"]
    youtube_list = current_settings["setup_variables"]["youtube_list"]
    merge_distance = current_settings["setup_variables"]["merge_distance"]
    menu(max_tokens, ai_model, transcribing_model, user_query, youtube_list, merge_distance)
    skip = input("Change Settings? (Y/n): ").strip().lower()
    if skip in ["y", "yes"]:
        while True:
            menu(max_tokens, ai_model, transcribing_model, user_query, youtube_list, merge_distance)

            choice = input("Input: ").strip()
            if choice == "1":
                manual = input("Do you want to manually enter max tokens (Y/n): ").strip().lower()
                if manual in ["y", "yes"]:
                    try:
                        raw_max = int(input("Enter new Max Tokens (total model limit): ").strip())
                        overhead = (
                            return_tokens(current_settings["system_variables"]["AI_instruction"]) +
                            return_tokens(current_settings["setup_variables"]["user_query"])
                        )
                        max_tokens = (raw_max - overhead)*0.6
                    except Exception as e:
                        print(f"Make sure it's an integer: {e}")
                else:
                    try:
                        raw_max = max_tokens_ai_check(ai_model)
                        overhead = (
                            return_tokens(current_settings["system_variables"]["AI_instruction"]) +
                            return_tokens(user_query)
                        )
                        max_tokens = (raw_max - overhead)*0.6
                        print(f"AI can handle up to {max_tokens} tokens for transcript content.")
                    except Exception as e:
                        print(f"AI at its task: {e}")
                        
            elif choice == "2":
                ai_model = input("Enter new AI Model: ")
            elif choice == "3":
                transcribing_model = input("Please enter the transcribing model name (e.g.,'tiny', 'base', 'small', 'medium', 'large'): ")
            elif choice == "4":
                user_query = input("Enter new User Query: ")
            elif choice == "5":
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
            elif choice == "7":
                try:
                    merge_distance = int(input("Enter new Merge Distance (highly impacts length of video): ").strip())
                except Exception as e:
                    print(f"Make sure its an integer: {e}")
            elif choice == "0":
                break
            else:
                print("Invalid choice. Please try again.")
            #Save updated settings
            new_settings = load(SETTINGS_FILE)
            new_settings["setup_variables"]["max_tokens"] = max_tokens
            new_settings["setup_variables"]["ai_model"] = ai_model
            new_settings["setup_variables"]["transcribing_model"] = transcribing_model
            new_settings["setup_variables"]["user_query"] = user_query
            new_settings["setup_variables"]["youtube_list"] = youtube_list
            new_settings["setup_variables"]["merge_distance"] = merge_distance
            wright(SETTINGS_FILE, new_settings)
            issue_checker(new_settings)

                

    print("Booting up...")
    settings = load(SETTINGS_FILE)
    ai_model = settings["setup_variables"]["ai_model"]
    try:
        interact_w_ai(ai_model)
        main_run = True
    except Exception as e:
        print(f"Error interacting with AI (Make sure Ollama is running): {e}")
    
    return main_run