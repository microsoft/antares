#!/bin/bash -e

cd $(dirname $0)/..
WORKDIR=$(pwd)

# Check Python Version
${PYTHON_EXEC:-python3} -c 'import sys; assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "Python Error: Antares depends on Python >= 3.6"'

if ${PYTHON_EXEC:-python3} -c 'assert "-packages/antares_core" in "'${WORKDIR}'"' >/dev/null 2>&1; then
    exit 0
fi

# Check Symlink Attributions
if [[ -e "engine/device-stub" ]] && [[ ! -L "engine/device-stub/lib64" ]]; then
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
		echo 'Error: The path of Antares ("'${WORKDIR}'") is put in a location that is invisible to Windows Host.'
		echo 'Please move this folder "'${WORKDIR}'" to a Windows-visible path, like: D:\xxx\..'
		echo
		exit 1
	fi
fi
