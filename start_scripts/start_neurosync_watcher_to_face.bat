@echo off
TITLE Neurosync Watcher To Face
cd /d "%~dp0..\Neurosync\NeuroSync_Player"
call %USERPROFILE%\miniconda30\Scripts\activate.bat
call conda activate mcp_env_311
python watcher_to_face.py
cmd /k