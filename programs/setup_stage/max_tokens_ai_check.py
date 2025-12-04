def max_tokens_ai_check(base_url: str, ai_model: str, prompt: str):
    from programs.components import return_tokens
    from openai import OpenAI
    
    def ask_ai(base_url: str, ai_model: str, prompt: str):
        client = OpenAI(base_url=base_url, api_key="not-needed")

        messages = [
            {"role": "user","content": (prompt)}
        ]

        response = client.chat.completions.create(
            model=ai_model,
            messages=messages,
            temperature=0,
        )
        return response

    #Figuring out how manny tokens the ai can handle
    prompt = "Hello today is a great day." * 100  #Starting prompt
    aggresivness = 100  #How many tokens to increase each time
    print("Starting max tokens AI check... with initial prompt length of", return_tokens(prompt), "tokens.")
    while True:
        tokens = return_tokens(prompt)

        try:
            response = ask_ai(base_url, ai_model, prompt)
            print(f"AI handled {tokens} tokens successfully. Increasing token count...")
            prompt += " Adding more text to increase token count. " * aggresivness

        except Exception as e:
            print(f"AI failed to respond with {tokens} tokens. Error: {e}")
            while True:
                aggresivness = 10
                prompt -= " Adding more text to increase token count. " * aggresivness
                try:
                    response = ask_ai(base_url, ai_model, prompt)
                    print(f"AI handled {return_tokens(prompt)} tokens successfully after decreasing.")
                    break
                except Exception as e:
                    print(f"AI still failed with {return_tokens(prompt)} tokens. Error: {e}")
            
            max_tokens = return_tokens(prompt)
            print("Max tokens the AI can handle is approximately:", max_tokens)
            break
    
    return max_tokens