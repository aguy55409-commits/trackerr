@echo off
cd /d "%~dp0dist\FN AL Tracker"
powershell -Command "Start-Process -FilePath '.\FN AL Tracker.exe' -Verb RunAs"
