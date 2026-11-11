@echo off
setlocal

:: name of the current folder
for %%I in ("%~dp0.") do set "FOLDER_NAME=%%~nxI"

:: server details
set "REMOTE_USER=group19@rockhill.csis.ul.ie"
set "REMOTE_DIR=/home/group19/%FOLDER_NAME%"

:: folders/files to upload and download
set "UPLOAD_FILES=src GUIDE.md README.md requirements.txt"
set "DOWNLOAD_FILES=results models"

:: options menu
echo Destination: %REMOTE_USER%:%REMOTE_DIR%/
echo 1. Upload: %UPLOAD_FILES%
echo 2. Download: %DOWNLOAD_FILES%
set /p CHOICE="Choose an option: "

:: execute chosen option
if "%CHOICE%"=="1" for %%F in (%UPLOAD_FILES%) do scp -r "%~dp0%%F" "%REMOTE_USER%:%REMOTE_DIR%/"
if "%CHOICE%"=="2" for %%F in (%DOWNLOAD_FILES%) do scp -r "%REMOTE_USER%:%REMOTE_DIR%/%%F" "%~dp0."

pause
