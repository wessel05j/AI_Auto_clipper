def variable_checker(user_query, base_url, model, clips_input, clips_output, transcribing_model, accuracy_model, max_token, youtube_list):
    value_errors = []

    if not user_query:
        value_errors.append("User query needs text")
    if not base_url:
        value_errors.append("base url needs to have text")
    if not model:
        value_errors.append("Model needs to have text")
    if not clips_input:
        value_errors.append("Clips input needs to have text")
    if not clips_output:
        value_errors.append("Clips output needs to have text")
    if not transcribing_model:
        value_errors.append("Transcribing model needs to have text")
    if not accuracy_model:
        value_errors.append("Accuracy model needs to have text")
    if max_token is None or not isinstance(max_token, int):
        value_errors.append("Max token needs an integer")
    if youtube_list is None or not isinstance(youtube_list, list):
        value_errors.append("Youtube list needs to be a list")

    
    return value_errors
    