def file_exists(filepath):
    import os
    """Check if a file exists and is not empty."""
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0