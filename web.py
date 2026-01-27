from flask import Flask, render_template_string
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DIR = os.path.join(BASE_DIR, "system")
STATUS_FILE = os.path.join(SYSTEM_DIR, "status.json")

@app.route('/')
def dashboard():
    status = {"current_video": "None", "progress": 0, "total_videos": 0, "current_step": "Idle"}
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
        except:
            pass
    html = f"""
    <html>
    <head><title>AI Auto Clipper Dashboard</title></head>
    <body>
        <h1>AI Auto Clipper Progress</h1>
        <p>Current Video: {status.get('current_video', 'None')}</p>
        <p>Progress: {status.get('progress', 0)}%</p>
        <p>Total Videos: {status.get('total_videos', 0)}</p>
        <p>Current Step: {status.get('current_step', 'Idle')}</p>
        <script>
            setTimeout(function(){{ location.reload(); }}, 2000);
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)