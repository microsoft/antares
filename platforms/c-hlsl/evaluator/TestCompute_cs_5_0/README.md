# How to evaluate and tuning kernels written in HLSL shader

1) First, you need a Win10 (64-bit) OS with DirectX12 enabled, and this host is used to activate the Antares HLSL Agent.

2) Then, you also need a Linux environment with docker service, which is used to launch the main Antares engine. Ensure the Linux environment and Win10 host can interact with each other via network.

3) Launch the Antares HLSL Agent on Win10 OS: Just by executing `eval_agent.bat` and waiting for the service daemon to start up successfully.

4) In the Linux environment, run a single test with Antares engine using command `AGENT_URL=<win10-ip-addr> BACKEND=c-hlsl make`, and this is to test whether the main Antares engine can connect and utilize remote HLSL Agent on Win10.

5) For tuning workloads, just run standard Antares tuning commands with just an additional environment variable `AGENT_URL` in Linux side, e.g.:

```sh
  AGENT_URL=<win10-ip-addr> STEP=50 COMPUTE_V1='<some-expression>' make
```
