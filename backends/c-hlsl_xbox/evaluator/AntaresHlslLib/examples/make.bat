@echo off

cd %~dp0

if exist antares_hlsl_v0.2dev0_x64.dll (
  echo Using local dependencies ..
) else (
  echo Downloading dependencies ..
  curl.exe -LOs https://github.com/microsoft/antares/releases/download/v0.1.0/antares_hlsl_v0.2dev0_x64.dll

  if errorlevel 1 (
    echo "`curl.exe` command not found in Windows PATH. Failed to download dependencies."
    pause
    exit
  )
)

echo Compiling helloworld example ..
C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe helloworld.cs

echo Compiling finished!
pause

echo Executing program ..
helloworld

echo Program finished!
pause
