def interact_w_ai(ai_model: str):
    import ollama

    messages = [
        {"role": "system", "content": "You are a api connection checker"},
        {"role": "user", "content": "return nothing"}
    ]

    response = ollama.chat(
        model=ai_model,
        messages=messages,
        options={"temperature": 0.1}
    )

    return response['message']['content']
