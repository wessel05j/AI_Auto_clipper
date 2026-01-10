def ai_clipping(transcribed_text, user_query, model, chunked_transcribed_text, system_message):
    import ollama
    import json
    import sys

    #Do not change this!:
    if len(chunked_transcribed_text) == 1:
        system_message = system_message
    else:
        system_message += '''
            Chunking:
            - You may only see part of the transcript.
            - Avoid starting/ending at obvious mid-thought edges of a chunk.
            - Never create a clip that uses the very last segment of a transcript chunk.
            - If the topic seems to continue beyond what you see, end at the last natural boundary available, not mid-sentence.'''

    # Provide the transcript in the exact JSON format described in the system message
    transcript_json = json.dumps(transcribed_text, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Transcript JSON: {transcript_json}"},
        {"role": "user", "content": f"User query: {user_query}"},
    ]

    print(f"\n[AI] Scanning chunk with {model}...")
    stream = ollama.chat(
        model=model,
        messages=messages,
        think='low',
        options={"temperature": 0.7},
        stream=True
    )
    
    raw_output = ""
    for chunk in stream:
        content = chunk['message']['content']
        print(content, end='', flush=True)
        raw_output += content
    print("\n")

    raw_output = raw_output.strip()

    # Clean up markdown code blocks if present
    if raw_output.startswith("```json"):
        raw_output = raw_output[7:]
    if raw_output.startswith("```"):
        raw_output = raw_output[3:]
    if raw_output.endswith("```"):
        raw_output = raw_output[:-3]
    
    raw_output = raw_output.strip()

    try:
        parsed_output = json.loads(raw_output)
    except json.JSONDecodeError:
        print("\n[WARNING] Failed to parse JSON from AI response.")
        print("This usually means the response was truncated because 'max_tokens' is too low or context limit was reached.")
        return []

    return parsed_output
