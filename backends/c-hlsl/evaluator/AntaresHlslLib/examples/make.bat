@echo off

echo Downloading dependencies ..
curl -LOs https://github.com/microsoft/antares/releases/download/v0.1.0/antares_hlsl_v0.1_x64.dll

echo Compiling helloworld example ..
C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe helloworld.cs

echo Compiling finished!
pause

echo Executing program ..
helloworld

echo Program finished!
pause
