@echo off
cd /d d:\AutoGame\AutoBangDream\bangcheater

call ndk-build

timeout /t 2 >nul

dir .\libs\x86_64\bangcheater >nul 2>&1
if %errorlevel% equ 0 (
    adb push .\libs\x86_64\bangcheater /data/local/tmp/bangcheater
    adb shell chmod 755 /data/local/tmp/bangcheater
)