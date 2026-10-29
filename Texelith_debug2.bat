@echo off
title Texelith Debug
cd /d "%~dp0"
py -3 Texelith.py > error.log 2>&1
if errorlevel 1 (
    echo Error occurred, check error.log
    notepad error.log
)
pause
