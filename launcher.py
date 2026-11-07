import os, sys, socket, subprocess, time

PORT = 8501
URL = f"http://localhost:{PORT}"
TITLE = "Student Performance Predictor"

if getattr(sys, "frozen", False):
    exe_dir = os.path.dirname(sys.executable)
    app_path = os.path.join(sys._MEIPASS, "app.py")
else:
    exe_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(exe_dir, "app.py")
os.chdir(exe_dir)   # user_data, regressor.pickle, data.csv are read from next to the .exe

# no console window: write messages to app_log.txt (send me this file if something fails)
try:
    _log = open(os.path.join(exe_dir, "app_log.txt"), "a", encoding="utf-8", buffering=1)
    sys.stdout = _log
    sys.stderr = _log
except Exception:
    pass


def port_open():
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def error_box(text):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text, TITLE, 0x10)
    except Exception:
        print(text)


# ---------------- mode 1: run the Streamlit server (started by mode 2) ----------------
if "--server" in sys.argv:
    import numpy, pandas, matplotlib   # makes PyInstaller bundle them
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", app_path,
                "--global.developmentMode=false",
                f"--server.port={PORT}",
                "--server.headless=true",
                "--browser.gatherUsageStats=false"]
    sys.exit(stcli.main())

# ---------------- mode 2: open the app in its own window ----------------
NO_WINDOW = 0x08000000   # CREATE_NO_WINDOW (Windows)
server = None
if not port_open():
    cmd = [sys.executable, "--server"] if getattr(sys, "frozen", False) \
        else [sys.executable, os.path.abspath(__file__), "--server"]
    server = subprocess.Popen(cmd, creationflags=NO_WINDOW, stdout=sys.stdout, stderr=sys.stderr)


def stop_server():
    if server is not None:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)],
                           creationflags=NO_WINDOW, capture_output=True)
        except Exception:
            server.kill()


for _ in range(240):                 # waits up to ~2 minutes for the server
    if port_open():
        break
    time.sleep(0.5)
else:
    stop_server()
    error_box("The app could not start. Please send the file app_log.txt (next to this program) to the developer.")
    sys.exit(1)

try:
    import webview
    webview.create_window(TITLE, URL, width=1200, height=820, min_size=(380, 600))
    webview.start()                  # returns when the window is closed
except Exception as e:
    print("window error:", repr(e))
    stop_server()
    error_box("Could not open the app window.\n\nInstall the free 'Microsoft Edge WebView2 Runtime' from Microsoft, then try again.")
    sys.exit(1)

stop_server()                        # closing the window also stops the app
sys.exit(0)