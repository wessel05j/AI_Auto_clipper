def chunking(transcribed_text, max_tokens):
    max_input = max_tokens*0.6
    chunked = []
    current_chunk = []
    tokens = 0
    for segment in transcribed_text:
        segment_tokens = len(segment[2]) + len(str(segment[1])) + len(str(segment[0]))

        if tokens + segment_tokens > max_input:
            chunked.append(current_chunk)
            current_chunk = []
            tokens = 0
        
        current_chunk.append([segment[0],segment[1],segment[2]])
        tokens += segment_tokens
    
    if current_chunk:
        chunked.append(current_chunk)
    
    return chunked