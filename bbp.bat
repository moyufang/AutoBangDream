@echo off
cd d:\AutoGame\AutoBangDream\bangcheater

call ndk-build

adb connect 127.0.0.1:7555
dir .\libs\x86_64\bangcheater >nul 2>&1
if %errorlevel% equ 0 (
    adb push .\libs\x86_64\bangcheater /data/local/tmp/bangcheater
    adb shell chmod 755 /data/local/tmp/bangcheater
)

cd ..