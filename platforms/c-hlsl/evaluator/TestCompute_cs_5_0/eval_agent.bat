@echo off

echo Compiling Eval Agent Service from Source ..
C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe AntaresHlslAgent.cs

if %errorlevel% neq 0 (pause) else (echo Waiting for Antares HLSL Eval Agent to Start up .. && powershell.exe Start-Process -Verb runas -FilePath AntaresHlslAgent.exe -ArgumentList server)
