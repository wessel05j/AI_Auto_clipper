from programs.core_functionality.ollama_chat import ollama_chat
import json

def ollama_scanning(transcribed_text, user_query, model, chunked_transcribed_text, system_message, temperature, max_tokens, url):
    '''Runs through a chunk of transcribed text and uses Ollama to find relevant clips.'''
    if len(chunked_transcribed_text) == 1:
        system_message = system_message
    else:
        system_message += '''
            Chunking Context:
            - You are viewing only ONE chunk of a larger transcript.
            - Segments at the chunk edges may be incomplete thoughts.
            - NEVER create clips using the final 2-3 segments of this chunk - they may continue into the next chunk.
            - ALWAYS end clips at clear sentence boundaries with proper punctuation (. ! ?) followed by a pause.
            - If context appears incomplete, end conservatively at the last confirmed complete sentence.'''

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