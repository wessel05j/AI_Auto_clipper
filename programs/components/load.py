def load(path):
    from programs.components.interact_w_json import interact_w_json
    settings = interact_w_json(path, "r", None)
    return settings