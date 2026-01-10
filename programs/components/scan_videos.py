def scan_videos(path):
    import os
    files = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            files.append(file_path)

    return files