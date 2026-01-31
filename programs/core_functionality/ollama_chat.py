from typing import Optional
import json
import requests
import logging

def ollama_chat(
    model: str,
    prompt: str,
    system_message: Optional[str] = None,
    temperature: float = 0.7,
    think: str = "low",
    stream: bool = False,
    max_output_tokens: int = 500,
    max_tokens: int = 4096,
    url: str = "http://localhost:11434",
) -> str:
    '''Send a chat prompt to an Ollama model using the official /api/chat contract.
    Args:
        model (str): The Ollama model to use (e.g., "llama2", "deepseek/r1").
        prompt (str): The user's prompt message.
        system_message (Optional[str]): An optional system message to set context.
        temperature (float): Sampling temperature for response generation.
        think (str): Level of "thinking" or deliberation by the model ("low", "medium", "high").
        stream (bool): Whether to stream the response or get it all at once.
        max_output_tokens (int): Maximum number of tokens to generate in the response.
        url (str): Base URL of the Ollama server.
    Returns:
        str: The model's response to the prompt.
    '''

    base_url = url.rstrip('/')
    if not base_url.endswith('/api'):
        api_url = f"{base_url}/api/chat"
    else:
        api_url = f"{base_url}/chat"
    
    payload = {
        "model": model,
        "messages": [],
        "stream": stream,
        # Ollama accepts model tuning values inside the options object.
        "options": {
            "temperature": temperature,
            "num_predict": max_output_tokens,
            "num_ctx": max_tokens,
            # Some models (e.g., Deepseek R1 style) honor a `think` option.
            "think": think,
        },
    }

    if system_message:
        payload["messages"].append({"role": "system", "content": system_message})

    payload["messages"].append({"role": "user", "content": prompt})

    headers = {"Content-Type": "application/json"}

    response = requests.post(api_url, headers=headers, data=json.dumps(payload), stream=stream)
    response.raise_for_status()

    if stream:
        # Aggregate streamed chunks into a single string.
        chunks = []
        for line in response.iter_lines():
            if not line:
                continue
            piece = json.loads(line)
            content = piece.get("message", {}).get("content")
            if content:
                chunks.append(content)
        return "".join(chunks)

    body = response.json()
    
    # Log the full response for debugging
    logging.debug(f"Ollama raw response: {json.dumps(body)}")
    
    # Extract content - handle both chat and completion formats
    content = ""
    if "message" in body and "content" in body["message"]:
        content = body["message"]["content"]
    elif "response" in body:
        content = body["response"]
    
    return content