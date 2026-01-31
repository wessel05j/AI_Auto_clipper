from programs.core_functionality.ollama_chat import ollama_chat
import json
import logging
from programs.components.return_tokens import return_tokens

def ollama_scanning(transcribed_text, user_query, model, chunked_transcribed_text, system_message, temperature, max_output_tokens, max_tokens, url):
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

    prompt = f"Transcript JSON: {transcribed_text}\nUser query: {user_query}"
    
    response = ollama_chat(
        model=model,
        prompt=prompt,
        system_message=system_message,
        temperature=temperature,
        stream=False,
        max_output_tokens=max_output_tokens,
        max_tokens=max_tokens,
        url=url,
    )

    responsed = response.strip()
    
    # Strip thinking model output
    if "</think>" in responsed:
        responsed = responsed.split("</think>")[-1].strip()

    try:
        parsed_output = json.loads(responsed)
        logging.debug(f"Parsed AI output: {parsed_output}")
        return parsed_output
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse AI response: {e}. Response was: {responsed}")
        return []
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse AI response: {e}. Response was: {responsed}")
        return []