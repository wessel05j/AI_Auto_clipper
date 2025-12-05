import tiktoken


def _get_encoder():
    """Return the single encoding used everywhere in this project.

    We intentionally always use `cl100k_base` so that:
    - `return_tokens` in `programs.components.return_tokens` and
    - this chunking logic

    count tokens in exactly the same way, avoiding mismatches between
    setup-time max token estimates and runtime chunk sizes.
    """
    return tiktoken.get_encoding("cl100k_base")


def chunking(transcribed_text, max_tokens, ai_model, response_reserve=200):
    max_tokens = int(max_tokens)
    response_reserve = int(response_reserve)

    enc = _get_encoder()
    max_input_tokens = max_tokens - response_reserve

    if max_input_tokens <= 0:
        raise ValueError("max_tokens must be greater than response_reserve")

    chunked = []
    current_chunk = []
    current_tokens = 0

    for segment in transcribed_text:
        start, end, text = segment

        # Encode full segment once
        full_segment_text = f"{start} {end} {text}"
        full_segment_tokens = enc.encode(full_segment_text)

        # If the full segment fits into an empty chunk, we can treat it as a unit
        if len(full_segment_tokens) <= max_input_tokens:
            segment_tokens = len(full_segment_tokens)

            # If it doesn't fit into the current chunk, flush and start a new one
            if current_tokens + segment_tokens > max_input_tokens:
                if current_chunk:
                    chunked.append(current_chunk)
                current_chunk = []
                current_tokens = 0

            current_chunk.append([start, end, text])
            current_tokens += segment_tokens
            continue

        # At this point, one segment is larger than max_input_tokens.
        # We split its *text* across multiple chunks to strictly honor the limit.

        # First, flush the current_chunk so we start fresh for this large segment.
        if current_chunk:
            chunked.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        # Tokenize only the text portion and then build pieces that fit.
        text_tokens = enc.encode(text)
        prefix_text = f"{start} {end} "
        prefix_tokens = enc.encode(prefix_text)
        prefix_len = len(prefix_tokens)

        # How many text tokens can we fit per chunk, accounting for prefix tokens
        if prefix_len >= max_input_tokens:
            raise ValueError(
                "Prefix (start/end) alone consumes the entire token budget; "
                "increase max_tokens or reduce reserved tokens."
            )

        max_text_tokens_per_chunk = max_input_tokens - prefix_len

        idx = 0
        n = len(text_tokens)
        while idx < n:
            # Take a slice of text tokens that fits within the remaining budget
            piece_tokens = text_tokens[idx : idx + max_text_tokens_per_chunk]
            idx += len(piece_tokens)

            # Decode just this piece and reconstruct the segment text
            piece_text = enc.decode(piece_tokens)
            current_chunk.append([start, end, piece_text])
            chunked.append(current_chunk)

            # Each piece starts a fresh chunk, since we are at the limit by design
            current_chunk = []
            current_tokens = 0

    # Final chunk if any remaining
    if current_chunk:
        chunked.append(current_chunk)

    return chunked
