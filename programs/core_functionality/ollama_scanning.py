from programs.core_functionality.ollama_chat import ollama_chat
import json
import logging

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
        url=url,
    )

    responsed = response.strip()
    
    # Strip thinking model output (e.g., "Thinking...\n...\n...done thinking.\n")
    if "...done thinking." in responsed.lower():
        responsed = responsed.split("...done thinking.")[-1].strip()
    elif "done thinking." in responsed.lower():
        responsed = responsed.split("done thinking.")[-1].strip()
    elif "</think>" in responsed.lower():
        responsed = responsed.split("</think>")[-1].strip()

    # Strip markdown code blocks
    if "```json" in responsed:
        responsed = responsed.split("```json")[-1]
    if "```" in responsed:
        responsed = responsed.split("```")[0]
    
    # Find the JSON array - extract text starting from first '[' to last ']'
    start_idx = responsed.find('[')
    end_idx = responsed.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        responsed = responsed[start_idx:end_idx + 1]
    
    responsed = responsed.strip()

    try:
        parsed_output = json.loads(responsed)
        return parsed_output
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse AI response: {e}. Response was: {responsed[:100]}")
        return []