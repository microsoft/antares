#!/bin/bash -xe

executor="$1"
kernel="$2"

[[ "$executor" == "" ]] && exit 1
[[ "$kernel" == "" ]] && exit 1

adb push $1 /sdcard/evaluator
adb push $kernel /sdcard/kernel.cc
adb shell "su -c 'pkill evaluator -9; mount -o rw,remount /system && cd /system && cp /sdcard/evaluator . && chmod a+x evaluator && ./evaluator /sdcard/kernel.cc'"
