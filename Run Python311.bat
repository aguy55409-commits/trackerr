@echo off
cd /d "%~dp0"
powershell -Command "Start-Process -FilePath 'C:\Users\Client\AppData\Local\Programs\Python\Python311\python.exe' -ArgumentList 'run.py' -WorkingDirectory '%~dp0' -Verb RunAs"
