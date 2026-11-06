import os
import sys
import time
import subprocess
import urllib.request

# Disable Streamlit development mode BEFORE importing Streamlit
os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# Find application directory
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
    RESOURCE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCE_DIR = BASE_DIR

APP_PATH = os.path.join(RESOURCE_DIR, "app.py")

URL = "http://127.0.0.1:8502"

# Streamlit child process
if "--streamlit-child" in sys.argv:

    from streamlit.web.cli import main

    sys.argv = [
        "streamlit",
        "run",
        APP_PATH,
        "--server.port=8502",
        "--server.address=127.0.0.1",
        "--server.headless=true",
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false"
    ]

    main()
    sys.exit(0)

# Import desktop window
import webview

# Start Streamlit
if getattr(sys, "frozen", False):
    command = [sys.executable, "--streamlit-child"]
else:
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--streamlit-child"
    ]

LOG_PATH = os.path.join(BASE_DIR, "launcher_error.log")

log_file = open(LOG_PATH, "w", encoding="utf-8")

server = subprocess.Popen(
    command,
    cwd=BASE_DIR,
    stdout=log_file,
    stderr=subprocess.STDOUT
)
try:
    # Wait for Streamlit server
    for i in range(90):

        if server.poll() is not None:
            raise RuntimeError("Streamlit server stopped unexpectedly")

        try:
            response = urllib.request.urlopen(URL, timeout=2)
            response.close()
            break

        except Exception:
            time.sleep(1)

    else:
        raise RuntimeError("Streamlit server failed to start")

    # Open desktop application
    webview.create_window(
        "Student Performance Predictor",
        URL,
        width=1200,
        height=800,
        min_size=(800, 600)
    )

    webview.start()

finally:
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()