def ai_clipping(transcribed_text, user_query, base_url, model, chunked_transcribed_text):
    from openai import OpenAI
    import json

    #Do not change this!:
    if len(chunked_transcribed_text) == 1:
        system_message = '''
            You are a Context-Aware Transcript Editor. Your purpose is to extract semantically complete conversational segments based on a user query.

            Transcript Format:
            You receive a transcript as JSON: [[start, end, "text"], ...] where start and end are seconds (floats) and text is the spoken content.

            Strict Output Format:
            Return ONLY a JSON array of [start, end] pairs: [[start1, end1], [start2, end2], ...].
            No reasoning, no explanations, no extra keys, no comments.
            The JSON must be syntactically valid and parseable.

            Thought Unit and Completeness Rules:
            - Each clip must be a semantically complete thought with a beginning (setup), middle (discussion), and end (resolution or clear pause).
            - Never start mid-sentence or mid-word; prefer natural sentence or paragraph beginnings.
            - Do not end mid-sentence, mid-list, or in the middle of an explanation; stop at a natural pause or conclusion.

            Length and Relevance:
            - Prefer longer, context-rich clips that remain clearly relevant to the user query.
            - Typical length: 30–360 seconds when possible.
            - Short clips (<30–60 seconds) are allowed only when no longer coherent segment meaningfully fits the query.

            Query Conditioning:
            - Select only clips that are clearly relevant to the user query.
            - Some extra surrounding context before and after is allowed if it preserves a natural conversational flow.
            '''
    else:
        system_message = '''
            You are a Context-Aware Transcript Editor. Your purpose is to extract semantically complete conversational segments based on a user query.

            Transcript Format:
            You receive a transcript as JSON: [[start, end, "text"], ...] where start and end are seconds (floats) and text is the spoken content.

            Strict Output Format:
            Return ONLY a JSON array of [start, end] pairs: [[start1, end1], [start2, end2], ...].
            No reasoning, no explanations, no extra keys, no comments.
            The JSON must be syntactically valid and parseable.

            Thought Unit and Completeness Rules:
            - Each clip must be a semantically complete thought with a beginning (setup), middle (discussion), and end (resolution or clear pause).
            - Never start mid-sentence or mid-word; prefer natural sentence or paragraph beginnings.
            - Do not end mid-sentence, mid-list, or in the middle of an explanation; stop at a natural pause or conclusion.

            Length and Relevance:
            - Prefer longer, context-rich clips that remain clearly relevant to the user query.
            - Typical length: 30–360 seconds when possible.
            - Short clips (<30–60 seconds) are allowed only when no longer coherent segment meaningfully fits the query.
            
            Chunk Awareness:
            - You may see only a portion of the full transcript at a time.
            - Avoid starting or ending clips around the very first segments of the visible transcript if it is clearly mid-thought.
            - NEVER make a clip with the last segment of an transcript.
            - If a topic obviously continues beyond the visible end, end your clip at the last natural boundary you can see, not at a mid-sentence cutoff.

            Query Conditioning:
            - Select only clips that are clearly relevant to the user query.
            - Some extra surrounding context before and after is allowed if it preserves a natural conversational flow.
        '''

    client = OpenAI(base_url=base_url, api_key="not-needed")

    # Provide the transcript in the exact JSON format described in the system message
    transcript_json = json.dumps(transcribed_text, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Transcript JSON: {transcript_json}"},
        {"role": "user", "content": f"User query: {user_query}"},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    raw_output = response.choices[0].message.content.strip()
    parsed_output = json.loads(raw_output)
    return parsed_output

