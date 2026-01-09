def ai_clipping(transcribed_text, user_query, model, chunked_transcribed_text, system_message):
    import ollama
    import json

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

    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0.7},
        think='medium'
    )

    raw_output = response['message']['content'].strip()
    parsed_output = json.loads(raw_output)
    return parsed_output

