def max_tokens_ai_check(base_url: str, ai_model: str):
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="not-needed")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How manny tokens can you handle in a single prompt and response? Only respond with the number."}
    ]

    response = client.chat.completions.create(
        model=ai_model,
        messages=messages,
        temperature=0
    )

    return int(response.choices[0].message.content)