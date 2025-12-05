def max_tokens_ai_check(base_url: str, ai_model: str):
    from programs.components.return_tokens import return_tokens
    from openai import OpenAI

    def ask_ai(base_url: str, ai_model: str, prompt: str):
        client = OpenAI(base_url=base_url, api_key="not-needed")
        messages = [
            {"role": "system", "content": "We are checking your maximum token capacity. Please respond with nothing."},
            {"role": "user", "content": prompt}
            ]
        return client.chat.completions.create(
            model=ai_model,
            messages=messages,
            temperature=0,
        )

    base_chunk = "Hello today is a great day."
    extra_chunk = " Adding more text to increase token count."
    chunk_count = 100
    step = 100

    prompt = base_chunk * chunk_count
    print("Starting max tokens AI check... with initial prompt length of", return_tokens(prompt), "tokens.")

    while True:
        tokens = return_tokens(prompt)
        try:
            ask_ai(base_url, ai_model, prompt)
            print(f"AI handled {tokens} tokens successfully. Increasing token count...")
            chunk_count += step
            prompt = base_chunk * chunk_count
        except Exception as e:
            print(f"AI failed to respond with {tokens} tokens")
            # back off
            while step > 1:
                step //= 2
                chunk_count -= step
                prompt = base_chunk * chunk_count
                try:
                    ask_ai(base_url, ai_model, prompt)
                    print(f"AI handled {return_tokens(prompt)} tokens successfully after decreasing.")
                    break
                except Exception as e2:
                    print(f"AI still failed with {return_tokens(prompt)} tokens. Error: {e2}")
            max_tokens = return_tokens(prompt)
            print("Max tokens the AI can handle is approximately:", max_tokens)
            return max_tokens