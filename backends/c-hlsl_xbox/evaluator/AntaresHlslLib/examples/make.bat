@echo off

cd %~dp0

if exist dxcompiler.dll (
  echo Using local dependencies ..
) else (
  echo Downloading dependencies: Microsoft DirectX Shader Compiler 6 ..
  curl.exe -Ls https://github.com/microsoft/antares/releases/download/v0.2.0/antares_hlsl_v0.3.3_x64.dll -o antares_hlsl_v0.3.3_x64.dll
  curl.exe -Ls https://github.com/microsoft/antares/releases/download/v0.2.0/dxcompiler.dll -o dxcompiler.dll
  curl.exe -Ls https://github.com/microsoft/antares/releases/download/v0.2.0/dxil.dll -o dxil.dll

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
