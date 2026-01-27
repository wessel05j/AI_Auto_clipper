from programs.core_functionality.ollama_chat import ollama_chat
import json

def ollama_scanning(transcribed_text, user_query, model, chunked_transcribed_text, system_message, temperature, max_tokens, url):
    '''Runs through a chunk of transcribed text and uses Ollama to find relevant clips.'''
    if len(chunked_transcribed_text) == 1:
        system_message = system_message
    else:
        system_message += '''
            Chunking:
            - You may only see part of the transcript.
            - Avoid starting/ending at obvious mid-thought edges of a chunk.
            - Never create a clip that uses the very last segment of a transcript chunk.
            - If the topic seems to continue beyond what you see, end at the last natural boundary available, not mid-sentence.'''

    response = ollama_chat(
        model=model,
        prompt=f"Transcript JSON: {transcribed_text}\nUser query: {user_query}",
        system_message=system_message,
        temperature=temperature,
        stream=False,
        max_tokens=max_tokens,
        url=url
    )

    responsed = response.strip()

    if responsed.startswith("```json"):
        responsed = responsed[7:]
    if responsed.startswith("```"):
        responsed = responsed[3:]
    if responsed.endswith("```"):
        responsed = responsed[:-3]

    parsed_output = json.loads(responsed)
    return parsed_output