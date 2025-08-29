@echo off
setlocal

set "SOURCE=%~1"
if "%SOURCE%"=="" set "SOURCE=%ANALYZER_SOURCE%"
if "%SOURCE%"=="" (
  echo Usage: %~nx0 ^<path_to_source^>
  echo Or set ANALYZER_SOURCE environment variable.
  exit /b 1
)
if not "%~1"=="" shift
python analyzer.py "%SOURCE%" %*
