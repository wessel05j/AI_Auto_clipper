def return_tokens(text):
    # Approximate token count (roughly 1 token per 4 characters in English)
    # This is a simple estimation that works reasonably well for Ollama models
    return len(text) // 4