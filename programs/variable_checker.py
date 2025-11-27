def variable_checker(user_query, base_url, model, clips_input, clips_output, transcribed_model, max_token):
    checker = []
    if not user_query:
        checker.append("User query needs text")
    if not base_url:
        checker.append("base url needs to have text")
    if not model:
        checker.append("Model needs to have text")
    if not clips_input:
        checker.append("Clips input needs to have text")
    if not clips_output:
        checker.append("Clips output needs to have text")
    if not transcribed_model:
        checker.append("Transcribed model needs to have text")
    if max_token is None or not isinstance(max_token, int):
        checker.append("Max token needs an integer")

    
    return checker
    