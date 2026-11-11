@echo off
setlocal

for %%I in ("%~dp0.") do set "FOLDER_NAME=%%~nxI"

set "REMOTE_USER=group19@rockhill.csis.ul.ie"
set "REMOTE_DIR=/home/group19/%FOLDER_NAME%"

set "FILES=src GUIDE.md README.md requirements.txt"

scp -r %FILES% "%REMOTE_USER%:%REMOTE_DIR%/"

pause
