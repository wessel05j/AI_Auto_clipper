def interact_w_json(path, w_or_r, data):
    import json
    
    if w_or_r.lower() == "w":
        with open(path, "w") as f:
            json.dump(data, f)
    
    elif w_or_r.lower() == "r":
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("w_or_r must be either 'w' for write or 'r' for read.")
