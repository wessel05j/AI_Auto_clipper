def max_tokens_ai_check(base_url: str, ai_model: str):
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="not-needed")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "How many tokens can you handle in a single prompt and response? "
                "Only respond with a plain integer number, no text or punctuation."
            ),
        },
    ]

    response = client.chat.completions.create(
        model=ai_model,
        messages=messages,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    # Remove common formatting like commas and spaces
    cleaned = raw.replace(",", "").strip()
    return int(cleaned)