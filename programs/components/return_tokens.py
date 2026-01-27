def return_tokens(text):
    import tiktoken
    # Use tiktoken for better token estimation (approximates for Ollama models)
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 style, reasonable for general use
        return len(encoding.encode(text))
    except:
        # Fallback to approximate count
        import math
        return math.floor(len(text) / 3.5)