title Normal Flip
cd /d "%~dp0"
where py >nul 2>nul && (py -3 normal_flip.py & goto :eof)
where python >nul 2>nul && (python normal_flip.py & goto :eof)
echo.
echo   Python not found. Install it from python.org and tick
echo   "Add python.exe to PATH" during setup, then run this file again.
echo.
pause
