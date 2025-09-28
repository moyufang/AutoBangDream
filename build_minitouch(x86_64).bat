cd libedev
ndk-build

adb push .\libs\x86_64\minitouch /data/local/tmp/bandcheater
adb shell chmod 755 /data/local/tmp/bandcheater