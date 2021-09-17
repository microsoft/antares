#!/bin/bash -e

cd $(dirname $0)/..

# Check Python Version
python3 -c 'import sys; assert sys.version >= "3.6", "Python Error: Antares depends on Python >= 3.6"'

# Check Symlink Attributions
if [[ ! -L "engine/device-stub/lib64" ]]; then
	echo
	echo 'Error: This repo is cloned with "Symbolic Link Attribution" lost.'
	echo 'Please re-clone this repo in true Linux environment, or Windows WSL environment, or using "Git for Windows" that supports keeping "Symbolic Link Attribution".'
	echo
	exit 1
fi

# Check Antares Location Correctness
if which fc.exe >/dev/null 2>&1; then
	if ! fc.exe README.md README.md >/dev/null 2>&1; then
		echo
		echo 'Error: The path of Antares ("'$(pwd)'") is put in a location that is invisible to Windows Host.'
		echo 'Please move this folder "'$(pwd)'" to a Windows-visible path, like: D:\xxx\..'
		echo
		exit 1
	fi
fi
