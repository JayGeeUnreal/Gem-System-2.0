@echo off
TITLE Neurosync Local API
cd /d "%~dp0..\Neurosync\NeuroSync_Local_API"
call %USERPROFILE%\miniconda30\Scripts\activate.bat
call conda activate mcp_env_311
python neurosync_local_api.py
cmd /k