@echo off
cd /d "%~dp0"
echo === Building StudentPredictor.exe ===
if exist .venv313\Scripts\activate.bat (call .venv313\Scripts\activate.bat) else (echo .venv313 not found - using system Python)
python -m pip install --upgrade pip
python -m pip install streamlit pyinstaller numpy pandas matplotlib tzdata pywebview
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
pyinstaller --noconfirm --onefile --noconsole --name StudentPredictor --collect-all streamlit --copy-metadata streamlit --collect-data tzdata --collect-all webview --hidden-import clr --hidden-import matplotlib.backends.backend_agg --add-data "app.py;." launcher.py
if not exist dist\StudentPredictor.exe (echo BUILD FAILED - copy the last lines above and send them to Claude & pause & exit /b 1)
copy /y regressor.pickle dist\ >nul
copy /y data.csv dist\ >nul
if exist .streamlit xcopy /e /i /y .streamlit dist\.streamlit >nul
if exist images xcopy /e /i /y images dist\images >nul
if exist image xcopy /e /i /y image dist\images >nul
echo.
echo DONE. Your app is dist\StudentPredictor.exe
pause