# Configure HLSL evaluator listener

DirectX12 is natively supported by WinOS. Since Antares works in LinuxOS, we use a remote tuning way to connect the evaluator deployed on a remote WinOS. Here is a tutorial on how to set up an HLSL evaluator listener on WinOS.

If this is the first time build on the target machine:

1) Compile [TestCompute](TestCompute.vcxproj) in Visual Studio

    It's recommended to compile the project with Visual Studio 2019. If the build is succeeded, you could find `TestCompute.exe` in your current folder. 

2) Register [TDR.reg](TDR.reg)

    To avoid the failure caused by TDR, You need to execute the registry to make the WinOS waiting for the GPU execution longer than 5s.

Start the listener:
``` 
python eval_agent.py
```

After the agent is started, the `Host_IP` and `Host_PORT` will be shown in the command line, which will be used in remote tuning requests by the client.
