import math

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")  # GPT-4 style, reasonable for general use
except Exception:
    _ENCODING = None


def return_tokens(text):
    # Use cached tokenizer when available for fast token estimation.
    text = str(text)
    if _ENCODING is not None:
        try:
            return len(_ENCODING.encode(text))
        except Exception:
            pass

    # Fallback to approximate count.
    return math.floor(len(text) / 3.5)
