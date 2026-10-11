@echo off
cd /d "%~dp0"
set "PY=%LocalAppData%\Programs\Python\Python311\python.exe"
if not exist "%PY%" set "PY=python"
powershell -NoProfile -Command "Start-Process -FilePath '%PY%' -ArgumentList 'run.py' -WorkingDirectory '%~dp0' -Verb RunAs"
if errorlevel 1 (
  echo Launch was cancelled or Python could not start.
  pause
)
