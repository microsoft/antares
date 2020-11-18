// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// This program is for `Win10-x64bit` only.
// [WSL CS Compiler] Using Command: C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe /out:helloworld.exe  helloworld.cs

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

using Microsoft.Win32;
using System.Runtime.CompilerServices;
using System.Threading;

namespace AntaresHelloWorldExample
{
    class Program
    {
        [DllImport(@"C:\Users\weicu\Desktop\antares\platforms\c-hlsl\evaluator\AntaresHlslLib\x64\Release\antares_hlsl.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dxLoadShader(string source, IntPtr num_outputs, IntPtr num_inputs);

        [DllImport(@"C:\Users\weicu\Desktop\antares\platforms\c-hlsl\evaluator\AntaresHlslLib\x64\Release\antares_hlsl.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dxAllocateBuffer(long bytes);

        [DllImport(@"C:\Users\weicu\Desktop\antares\platforms\c-hlsl\evaluator\AntaresHlslLib\x64\Release\antares_hlsl.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dxLaunchShader(IntPtr kptr, IntPtr[] source);

        [DllImport(@"C:\Users\weicu\Desktop\antares\platforms\c-hlsl\evaluator\AntaresHlslLib\x64\Release\antares_hlsl.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dxMemcpyHostToBuffer(IntPtr dptr, IntPtr hptr, long bytes);

        [DllImport(@"C:\Users\weicu\Desktop\antares\platforms\c-hlsl\evaluator\AntaresHlslLib\x64\Release\antares_hlsl.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dxMemcpyBufferToHost(IntPtr hptr, IntPtr dptr, long bytes);

        [DllImport(@"C:\Users\weicu\Desktop\antares\platforms\c-hlsl\evaluator\AntaresHlslLib\x64\Release\antares_hlsl.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void dxSynchronize();

        static void initRegistryForSafeTDR()
        {
            const int timeout = 70;
            try
            {
                using (RegistryKey key = Registry.LocalMachine.CreateSubKey(@"SYSTEM\CurrentControlSet\Control\GraphicsDrivers"))
                {
                    key.SetValue("TdrLevel", 3);
                    key.SetValue("TdrDelay", timeout);
                    key.SetValue("TdrDdiDelay", timeout);
                    key.SetValue("TdrTestMode", 0);
                    key.SetValue("TdrDebugMode", 2);
                    key.SetValue("TdrLimitTime", timeout);
                    key.SetValue("TdrLimitCount", 0x1000000);
                    key.Close();
                }
                using (RegistryKey key = Registry.LocalMachine.CreateSubKey(@"SYSTEM\CurrentControlSet\Control\GraphicsDrivers\DCI"))
                {
                    key.SetValue("Timeout", timeout);
                    key.Close();
                }
            }
            catch (UnauthorizedAccessException)
            {
                Console.Error.WriteLine("[WARNING] Failed to add safe TDR settings into Windows registry.");
                Console.Error.WriteLine("[WARNING] Super privilege is required for safe TDR settings.");
                Console.Error.WriteLine("[WARNING] Otherwrise, invalid shaders might trigger Win10's blue screen.");
                Console.Error.WriteLine();
            }
        }

        static void Main(string[] args)
        {
            initRegistryForSafeTDR();

            // HLSL source code generated by Antares HLSL backend
            var antares_code = @"///32-1024/float32/input0:32/float32/output0
// backend = c-hlsl
// CONFIG:
// COMPUTE_V1: - output0[N] +=! input0[N, C] ...

StructuredBuffer<float> input0: register(t0);
RWStructuredBuffer<float> output0: register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 threadIdx: SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uint3 dispatchIdx : SV_DispatchThreadID) {
    // [thread_extent] blockIdx.x = 32
    float output0_local[1];
    // [thread_extent] threadIdx.x = 1
    output0_local[(0)] = 0.000000e+00f;
    for (int rv_outer = 0; rv_outer < 1024; ++rv_outer) {
        output0_local[(0)] = (output0_local[(0)] + input0[(((((int)blockIdx.x) * 1024) + rv_outer))]);
    }
    output0[(((int)blockIdx.x))] = output0_local[(0)];
}";

            Console.WriteLine("Compiling shaders ..");
            var handle = dxLoadShader(antares_code, IntPtr.Zero, IntPtr.Zero);
            if (handle == IntPtr.Zero)
                throw new Exception("Invalid Shader Source for Compilation.");

            Console.WriteLine("Initializing buffers ..");
            var d_input0 = dxAllocateBuffer(32 * 1024 * sizeof(float)); // create device buffer for input0
            var d_output0 = dxAllocateBuffer(32 * sizeof(float)); // create device buffer for output0

            // fill input data
            var h_input0 = new float[32 * 1024];
            for (int i = 0; i < h_input0.Length; ++i)
              h_input0[i] = 1;
            dxMemcpyHostToBuffer(d_input0, Marshal.UnsafeAddrOfPinnedArrayElement(h_input0, 0), h_input0.Length * sizeof(float));

            Console.WriteLine("Executing shaders asynchronously ..");
            // execute hlsl shader
            var kargs = new IntPtr[]{d_input0, d_output0};
            dxLaunchShader(handle, kargs);
            dxSynchronize();

            // read output data
            var h_output0 = new float[32];
            dxMemcpyBufferToHost(Marshal.UnsafeAddrOfPinnedArrayElement(h_output0, 0), d_output0, h_output0.Length * sizeof(float));
            Console.WriteLine("Waiting for execution to complete ..");
            dxSynchronize();

            // print result
            Console.WriteLine("Reading results back:");
            Console.WriteLine("  output0 = [" + h_output0[0] + ", " + h_output0[1] + ", .., " + h_output0[31] + "]");
            Console.Write("Program finished successfully.");
            Console.ReadKey();
        }
    }
}
