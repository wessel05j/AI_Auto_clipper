def setup_stage(settings_path: str):
    #Python imports
    import os
    

    #Component imports
    from programs.components.interact_w_json import interact_w_json
    from programs.components.file_exists import file_exists
    from programs.setup_stage.interact_w_ai import interact_w_ai
    from programs.setup_stage.max_tokens_ai_check import max_tokens_ai_check

    #Variables
    setup_needed = False
    main_run = False

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
            },
            "accuracy_model": {
                "accuracy_testing": False,
                "accuracy_model": ""
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
            while not os.path.exists(output_folder):
                print("The specified output folder does not exist. Please try again.")
                output_folder = input("Please enter the output folder path: ")

        input_folder = input("Please enter the input folder path: ")
        if "/" not in input_folder and "\\" not in input_folder:
            input_folder = input_folder + "/"
        if not os.path.exists(input_folder):
            while not os.path.exists(input_folder):
                print("The specified input folder does not exist. Please try again.")
                input_folder = input("Please enter the input folder path: ")
        
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

        print("Finding out how many tokens the AI can handle...")
        while True:
            try:
                max_tokens = max_tokens_ai_check(base_url, ai_model)
                print(f"AI can handle up to {max_tokens} tokens per prompt and response.")
                break
            except Exception as e:
                print("Failed to determine max tokens from AI, Trying again... Ai model might be weak at this.")
                print("Error details: ", str(e))
        
        #Now creating your settings folder and file
        template_settings["setup_variables"]["max_tokens"] = max_tokens #int
        template_settings["setup_variables"]["output_folder"] = output_folder #str
        template_settings["setup_variables"]["input_folder"] = input_folder #str
        template_settings["setup_variables"]["ai_model"] = ai_model #str
        template_settings["setup_variables"]["base_url"] = base_url #str
        template_settings["setup_variables"]["transcribing_model"] = transcribing_model #str
        template_settings["setup_variables"]["user_query"] = user_query #str
        interact_w_json(settings_path, "w", template_settings)
    
    settings = interact_w_json(settings_path, "r", None)
    skip = input("Are you currently running a session and just want to skip booting stage? (y/N): ").strip().lower()
    if skip in ["n", "no"]:
        #Declaring external variables before checking
        if not settings["accuracy_model"]["accuracy_testing"]:
            accuracy_testing_input = input("Do you want to enable accuracy testing? This will re-transcribe the clips at the end to check for accuracy. (Y/n)").strip().lower()
            while accuracy_testing_input.lower() not in ["y", "n", "yes", "no"]:
                accuracy_testing_input = input("Invalid input. Please enter 'Y' for yes or 'N' for no: ").strip().lower()
            if accuracy_testing_input in ["y", "yes"]:
                accuracy_testing = True
                accuracy_model = input("Please enter the accuracy transcribing model name (e.g.,'tiny', 'base', 'small', 'medium', 'large'): ")
                transcribing_models = ["tiny", "base", "small", "medium", "large"]
                while accuracy_model not in transcribing_models:
                    accuracy_model = input(f"Invalid model name. Please choose from {transcribing_models}: ")
                settings["accuracy_model"]["accuracy_testing"] = accuracy_testing
                settings["accuracy_model"]["accuracy_model"] = accuracy_model
                interact_w_json(settings_path, "w", settings)
                settings = interact_w_json(settings_path, "r", None)
        
        youtube_list_input = input("Do you want to download videos from YouTube? If yes, please enter the links separated by commas. If no, just press Enter: ").strip()
        if youtube_list_input:
            youtube_list = [link.strip() for link in youtube_list_input.split(",")]
            settings["setup_variables"]["youtube_list"] = youtube_list
            interact_w_json(settings_path, "w", settings)
            settings = interact_w_json(settings_path, "r", None)

        #Checking the variables
        print("Checking your settings for potential issues...")
        value_errors = []
        if max_tokens <= 500:
            value_errors.append("Your AI max tokens setting is too low. Please choose higher capacity AI.")
        
        if not os.path.exists(output_folder):
            value_errors.append("The specified output folder does not exist: " + output_folder)
        
        if not os.path.exists(input_folder):
            value_errors.append("The specified input folder does not exist: " + input_folder)

        if ai_model.strip() == "":
            value_errors.append("The AI model is not set.")

        if base_url.strip() == "":
            value_errors.append("The AI base URL is not set.")
        
        if transcribing_model.strip() == "":
            value_errors.append("The transcribing model is not set.")
        
        if user_query.strip() == "":
            value_errors.append("The user query is not set.")
        
        if youtube_list is None:
            value_errors.append("The youtube list is not set.")
        
        if accuracy_testing:
            if accuracy_model.strip() == "":
                value_errors.append("The accuracy model is not set while accuracy testing is enabled.")
        print(f"Found {len(value_errors)} potential issues.\n")
        if len(value_errors) > 0:
            for error in value_errors:
                print("Potential issue: ", error)

        #Boot menu
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
                print(f"9. Accuracy Testing: {accuracy_testing}")
                if settings["accuracy_model"]["accuracy_testing"]:
                    print(f"10. Accuracy Model: {accuracy_model}")
                print("0. Done Editing")

                choice = input("Enter the number of your choice: ").strip()
                if choice == "1":
                    manual = input("Do you want to manually enter max tokens or let AI do it? (Y/n): ").strip().lower()
                    if manual in ["y", "yes"]:
                        try:
                            max_tokens = int(input("Enter new Max Tokens: ").strip())
                            print(f"AI can handle up to {max_tokens} tokens per prompt and response.")
                        except Exception as e:
                            print(f"Error interacting with AI (Make sure the AI service is available): {e}")
                    else:
                        max_tokens = max_tokens_ai_check(base_url, ai_model)
                        print(f"AI can handle up to {max_tokens} tokens per prompt and response.")
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
                    youtube_list_input = input("Enter new YouTube Links separated by commas: ").strip()
                    youtube_list = [link.strip() for link in youtube_list_input.split(",")]
                elif choice == "9":
                    accuracy_testing_input = input("Enable accuracy testing? (Y/n): ").strip().lower()
                    accuracy_testing = accuracy_testing_input in ["y", "yes"]
                elif choice == "10" and settings["accuracy_model"]["accuracy_testing"]:
                    accuracy_model = input("Enter new Accuracy Model: ")
                elif choice == "0":
                    break
                else:
                    print("Invalid choice. Please try again.")
            #Save updated settings
            updated_settings = {
                "setup_variables": {
                    "max_tokens": max_tokens,
                    "output_folder": output_folder,
                    "input_folder": input_folder,
                    "ai_model": ai_model,
                    "base_url": base_url,
                    "transcribing_model": transcribing_model,
                    "user_query": user_query,
                    "youtube_list": youtube_list,
                },
                "accuracy_model": {
                    "accuracy_testing": accuracy_testing,
                    "accuracy_model": accuracy_model if accuracy_testing else ""
                }
            }
            interact_w_json(settings_path, "w", updated_settings)
            settings = interact_w_json(settings_path, "r", None)

    print("Booting up...")
    settings = interact_w_json(settings_path, "r", None)
    base_url = settings["setup_variables"]["base_url"]
    ai_model = settings["setup_variables"]["ai_model"]
    try:
        interact_w_ai(base_url, ai_model)
        main_run = True
    except Exception as e:
        print(f"Error interacting with AI (Make sure the AI service is available): {e}")
    
    return main_run
            
        
    
    

    