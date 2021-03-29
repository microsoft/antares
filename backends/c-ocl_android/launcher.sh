#!/bin/bash -xe

executor="$1"
kernel="$2"

[[ "$executor" == "" ]] && exit 1
[[ "$kernel" == "" ]] && exit 1

exec_name=$(basename "$executor")
kernel_name=$(basename "$kernel")

adb push $1 /sdcard/$exec_name
adb push $kernel /sdcard/$kernel
adb shell "su -c 'cd /system && cp /sdcard/$exec_name . && chmod a+x $exec_name && ./$exec_name /sdcard/$kernel'"
