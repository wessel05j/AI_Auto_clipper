def interact_w_ai(base_url: str, ai_model: str):
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="not-needed")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Are we connected?"}
    ]

    response = client.chat.completions.create(
        model=ai_model,
        messages=messages
    )

    return response.choices[0].message.content
