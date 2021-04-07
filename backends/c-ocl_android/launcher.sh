#!/bin/bash -xe

executor="$1"
kernel="$2"

[[ "$executor" == "" ]] && exit 1
[[ "$kernel" == "" ]] && exit 1

cat "$1" | adb shell "su -c 'pkill evaluator -9; mount -o rw,remount /system; dd of=/system/evaluator && chmod a+x /system/evaluator'"
cat "$kernel" | adb shell "su -c 'dd of=/system/kernel.cc && /system/evaluator /system/kernel.cc'"
