@echo off
title LLM translator Launcher
echo Starting LLM translator...
echo.

call .\.venv\Scripts\activate

:: Run the application
python app.py

pause
