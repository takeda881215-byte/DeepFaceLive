@echo off
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
python launch_deepfacelive.py
pause
