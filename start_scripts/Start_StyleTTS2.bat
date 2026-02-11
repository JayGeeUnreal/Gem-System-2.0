@echo off
TITLE StyleTTS2
cd /d "%~dp0..\StyleTTS2"
call %USERPROFILE%\miniconda30\Scripts\activate.bat
call conda activate mcp_env_311
call python watcher.py
cmd /k