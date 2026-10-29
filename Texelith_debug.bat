title Texelith Debug
cd /d "%~dp0"
where py >nul 2>nul && (py -3 Texelith.py 2>&1 | tee error.log & pause & goto :eof)
where python >nul 2>nul && (python Texelith.py 2>&1 | tee error.log & pause & goto :eof)
echo.
echo   Python not found. Install it from python.org and tick
echo   "Add python.exe to PATH" during setup, then run this file again.
echo.
pause
