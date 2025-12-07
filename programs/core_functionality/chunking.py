def chunking(transcribed_text, max_tokens, safety_range=100) -> list:
    import tiktoken
    max_tokens = int(max_tokens)
    response_reserve = int(response_reserve)

    enc = tiktoken.get_encoding("cl100k_base")

    chunked = []
    current_chunk = []
    current_tokens = 0

    for segment in transcribed_text:
        start, end, text = segment

        # Encode full segment once
        full_segment_text = f"{start} {end} {text}"
        full_segment_tokens = enc.encode(full_segment_text)

        # If the full segment fits into an empty chunk, we can treat it as a unit
        if len(full_segment_tokens) <= max_tokens - safety_range:
            segment_tokens = len(full_segment_tokens)

            # If it doesn't fit into the current chunk, flush and start a new one
            if current_tokens + segment_tokens > max_tokens - safety_range:
                if current_chunk:
                    chunked.append(current_chunk)
                current_chunk = []
                current_tokens = 0

            current_chunk.append([start, end, text])
            current_tokens += segment_tokens
            continue

    if current_chunk:
        chunked.append(current_chunk)

    return chunked
