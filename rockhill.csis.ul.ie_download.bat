@echo off
setlocal

for %%I in ("%~dp0.") do set "FOLDER_NAME=%%~nxI"

set "REMOTE_USER=group19@rockhill.csis.ul.ie"
set "REMOTE_DIR=/home/group19/%FOLDER_NAME%"

set "FILES=results models"

scp -r "%REMOTE_USER%:%REMOTE_DIR%/%FILES%" "%~dp0."

pause
