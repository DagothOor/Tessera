@echo off
title Texelith GPU test
cd /d "%~dp0"
where py >nul 2>nul && (py -3 gpu_test.py & goto :done)
python gpu_test.py
:done
echo.
pause
