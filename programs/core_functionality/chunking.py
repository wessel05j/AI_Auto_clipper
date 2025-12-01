def chunking(transcribed_text, max_tokens, response_reserve=200):
    import tiktoken  # or the tokenizer your model uses
    """
    Splits transcribed_text into chunks based on real LLM tokens.
    
    transcribed_text: list of [start, end, text]
    max_tokens: model max (e.g., 16_000)
    response_reserve: tokens you want to leave for model output
    """
    
    enc = tiktoken.encoding_for_model("gpt-4o-mini")  # use your model's tokenizer
    max_input_tokens = int(max_tokens - response_reserve)

    chunked = []
    current_chunk = []
    current_tokens = 0

    for segment in transcribed_text:
        # Segment fields
        start, end, text = segment

        # Count actual prompt tokens
        segment_text = f"{start} {end} {text}"
        segment_tokens = len(enc.encode(segment_text))

        # If adding this segment would exceed the input limit → start new chunk
        if current_tokens + segment_tokens > max_input_tokens:
            if current_chunk:
                chunked.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        # Add segment
        current_chunk.append([start, end, text])
        current_tokens += segment_tokens

    # Final chunk
    if current_chunk:
        chunked.append(current_chunk)

    return chunked
