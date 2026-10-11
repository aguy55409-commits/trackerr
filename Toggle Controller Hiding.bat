@echo off
REM Hides/unhides the real Xbox pad from games so only the virtual DS4 is seen.
REM While hiding is ON your controller will NOT work in games unless the
REM tracker is running, because the tracker is what passes input through.

set CLI="C:\Program Files\Nefarius Software Solutions\HidHide\x64\HidHideCLI.exe"

if not exist %CLI% (
    echo HidHide is not installed at the expected path.
    pause
    exit /b 1
)

%CLI% --cloak-state | find /i "cloak-on" >nul
if %errorlevel%==0 (
    %CLI% --cloak-off
    echo.
    echo   Controller hiding is now OFF.
    echo   Your Xbox pad works normally in all games. No aim assist.
) else (
    %CLI% --cloak-on
    echo.
    echo   Controller hiding is now ON.
    echo   Games only see the virtual DS4, so run the tracker for input
    echo   to reach the game at all.
)
echo.
echo   ------------------------------------------------------------
echo    IMPORTANT: close the game BEFORE toggling, and start it
echo    afterwards. A game that is already running has the pad
echo    open and keeps reading it, so toggling now will not
echo    affect it until the game is restarted.
echo   ------------------------------------------------------------
echo.
pause
