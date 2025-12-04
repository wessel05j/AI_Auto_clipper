def return_tokens(text):
    import tiktoken

    #Choose encoding
    encoding = tiktoken.get_encoding("cl100k_base")

    #Encode the text
    tokenized_text = encoding.encode(text)

    #Return the number of tokens
    return len(tokenized_text)