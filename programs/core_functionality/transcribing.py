def transcribe_video(folder, model_name, merge_seconds=30, min_pause=0.3):
    import whisper, re

    # Load model
    model = whisper.load_model(model_name)
    result = model.transcribe(folder, verbose=False)

    # Regex for strong sentence ending
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

        # --- RULES FOR SAFE MERGING ---

        duration = current["end"] - current["start"]
        pause = seg["start"] - current["end"]

        has_strong_end = bool(strong_end.search(current["text"]))
        is_short = duration < merge_seconds
        pause_too_small = pause < min_pause  # avoid merging across big silence

        # Merge conditions:
        # 1. Current segment is short
        # 2. Text does NOT end with a strong end
        # 3. No big pause between segments
        if is_short and not has_strong_end and pause_too_small:
            current["end"] = seg["end"]
            current["text"] = (current["text"] + " " + seg_text).strip()
        else:
            # finalize
            merged.append([
                current["start"],
                current["end"],
                current["text"].strip()
            ])

            # start new
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg_text
            }

    # Append last one
    if current:
        merged.append([current["start"], current["end"], current["text"].strip()])

    return merged
