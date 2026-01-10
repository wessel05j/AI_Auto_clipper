def chunking(transcribed_text, max_tokens, safety_range=300) -> list:
    from programs.components.return_tokens import return_tokens
    max_tokens = int(max_tokens)

    chunked = []
    current_chunk = []
    current_tokens = 0

    for segment in transcribed_text:
        start, end, text = segment

        # Encode full segment once (approximate token count)
        full_segment_text = f"{start} {end} {text}"
        full_segment_tokens = return_tokens(full_segment_text)

        # If the full segment fits into an empty chunk, we can treat it as a unit
        if full_segment_tokens <= max_tokens - safety_range:
            segment_tokens = full_segment_tokens

            # If it doesn't fit into the current chunk, flush and start a new one
            if current_tokens + segment_tokens > max_tokens - safety_range:
                if current_chunk:
                    chunked.append(current_chunk)
                current_chunk = []
                current_tokens = 0

            current_chunk.append([start, end, text])
            current_tokens += segment_tokens
        else:
            # Segment is too large - split it into smaller pieces
            words = text.split()
            if not words:
                continue
            
            words_per_second = len(words) / max(1, end - start)
            target_words = max(1, int((max_tokens - safety_range) * 0.8))  # Conservative estimate
            
            for i in range(0, len(words), target_words):
                chunk_words = words[i:i + target_words]
                chunk_text = " ".join(chunk_words)
                chunk_start = start + (i / words_per_second) if words_per_second > 0 else start
                chunk_end = start + ((i + len(chunk_words)) / words_per_second) if words_per_second > 0 else end
                
                chunk_segment_text = f"{chunk_start} {chunk_end} {chunk_text}"
                chunk_tokens = return_tokens(chunk_segment_text)
                
                if current_tokens + chunk_tokens > max_tokens - safety_range:
                    if current_chunk:
                        chunked.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                
                current_chunk.append([chunk_start, chunk_end, chunk_text])
                current_tokens += chunk_tokens

    if current_chunk:
        chunked.append(current_chunk)

    return chunked
