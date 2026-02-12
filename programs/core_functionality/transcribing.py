def transcribe_video(folder, model, min_pause=3.0):
    import re
    import logging

    try:
        # Use provided model
        result = model.transcribe(folder, verbose=False)
    except Exception as e:
        logging.error(f"Failed to load audio: {e}")
        raise

    # Regex for strong sentence ending (includes quotes and spaces)
    strong_end = re.compile(r"[\.!\?]+(?:\"|'|\s|$)")

    merged = []
    current = None

    for seg in result["segments"]:
        seg_text = seg["text"].strip()

        if current is None:
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg_text
            }
            continue

        # Calculate the silence gap between segments
        pause = seg["start"] - current["end"]

        # Check if the PREVIOUS chunk ended a sentence
        has_strong_end = bool(strong_end.search(current["text"]))
        
        # Logic: If the sentence is finished OR they paused for > 3 seconds, we cut.
        # Otherwise, we keep building the sentence.
        if has_strong_end or pause >= min_pause:
            # finalize the completed sentence/thought
            merged.append([
                current["start"],
                current["end"],
                current["text"].strip()
            ])

            # start a brand new chunk
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg_text
            }
        else:
            # Still in the same thought, extend the end time and append text
            current["end"] = seg["end"]
            current["text"] = (current["text"] + " " + seg_text).strip()

    # Append the final remaining piece
    if current:
        merged.append([current["start"], current["end"], current["text"].strip()])

    return merged