def transcribe_video(folder, model):
    import whisper
    model = whisper.load_model(model)
    result = model.transcribe(folder, verbose=False)
    merged = []
    current = None
    for seg in result["segments"]:
        if current is None:
            current = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            continue

        # If current merged duration is still short and text doesn't end with .?!, keep merging
        too_short = current["end"] - current["start"] < 30
        ends_clean = current["text"].strip().endswith((".", "?", "!"))
        if too_short and not ends_clean:
            current["end"] = seg["end"]
            current["text"] += " " + seg["text"]
        else:
            merged.append([current["start"], current["end"], current["text"]])
            current = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}

    if current:
        merged.append([current["start"], current["end"], current["text"]])

    return merged
