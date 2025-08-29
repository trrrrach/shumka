@echo off
setlocal
cd /d %~dp0
if "%~1"=="" (
    echo Drag and drop a folder or ZIP/RAR archive onto this script,
    echo or enter a path below.
    set /p SOURCE=Source path: 
) else (
    set "SOURCE=%~1"
)
python analyzer.py "%SOURCE%"
pause
