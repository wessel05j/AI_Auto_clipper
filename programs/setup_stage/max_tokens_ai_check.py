def ask_ai(ai_model: str):
    import ollama
    messages = [
        {"role": "system", "content": "We are checking your maximum token capacity"},
        {"role": "user", "content": "Please respond with only how many tokens you can handle."}
        ]
    response = ollama.chat(
        model=ai_model,
        messages=messages,
        options={"temperature": 0.1}
    )
    return int(response['message']['content'])