#!/bin/bash -xe

executor="$1"
kernel="$2"

[[ "$executor" == "" ]] && exit 1
[[ "$kernel" == "" ]] && exit 1

if [[ "$BACKEND" == "c-mcpu_android" ]]; then
  aarch64-linux-android-clang++ $kernel -D__BACKEND_mcpu_android__ -std=c++17 -Wno-string-compare -Wno-unused-result -Wno-unused-value -o .libcpu_module.so -ldl -O3 -shared
  cat .libcpu_module.so | adb shell "su -c 'pkill evaluator -9; mount -o rw,remount /system; dd of=/system/libcpu_module.so && chmod a+x /system/libcpu_module.so'"
  rm -f .libcpu_module.so
fi

cat "$1" | adb shell "su -c 'pkill evaluator -9; mount -o rw,remount /system; dd of=/system/evaluator && chmod a+x /system/evaluator'"
cat "$kernel" | timeout 30 adb shell "su -c 'dd of=/system/kernel.cc && CPU_THREADS=${CPU_THREADS} /system/evaluator /system/kernel.cc'"
