import threading
import time
import webview
import subprocess
import sys
import os
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(BASE_DIR, "app.py")

def run_streamlit():
    subprocess.Popen([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        APP_PATH,
        "--server.port=8501",
        "--server.headless=true"
    ], cwd=BASE_DIR)

threading.Thread(target=run_streamlit, daemon=True).start()

url = "http://localhost:8501"

for i in range(60):
    try:
        urllib.request.urlopen(url, timeout=1)
        break
    except Exception:
        time.sleep(1)

webview.create_window(
    "Study Hours & Exam Marks",
    url,
    width=1200,
    height=800
)

webview.start()