// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// This program is for `Win10-x64bit` only.
// [WSL CS Compiler] Using Command: C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe /out:program.exe program.cs

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Threading;

using Microsoft.Win32;
using System.Diagnostics;

namespace AntaresHelloWorldExample
{
    class Program
    {
        public const string HlslDllName = @"antares_hlsl_v0.1_x64.dll";

        [DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxInit(int flags);

        [DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxStreamSynchronize(IntPtr hStream);

        [DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dxShaderLoad(string source, [Optional] int num_outputs, [Optional] int num_inputs);
        [DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dxShaderLoad(string source, out int num_outputs, out int num_inputs);

        [DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxShaderLaunchAsync(IntPtr hShader, IntPtr[] source, IntPtr hStream);

        [DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dxMemAlloc(long bytes);
        
        [DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxMemcpyHtoDAsync(IntPtr dptr, IntPtr hptr, long bytes, IntPtr hStream);

        [DllImport(HlslDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxMemcpyDtoHAsync(IntPtr hptr, IntPtr dptr, long bytes, IntPtr hStream);

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool FreeLibrary(IntPtr hModule);

        public static void UnloadHlslImportedDll()
        {
            foreach (System.Diagnostics.ProcessModule mod in System.Diagnostics.Process.GetCurrentProcess().Modules)
            {
                if (mod.ModuleName == HlslDllName)
                {
                    FreeLibrary(mod.BaseAddress);
                }
            }
        }

        static int Main(string[] args)
        {

            // HLSL source code generated by Antares HLSL backend
            var antares_code = @"///524288/float32/input0,524288/float32/input1:524288/float32/output0
// BACKEND = c-hlsl
// ...

StructuredBuffer<float> input0: register(t0);
StructuredBuffer<float> input1: register(t1);
RWStructuredBuffer<float> output0: register(u0);


[numthreads(32, 1, 1)]
void CSMain(uint3 threadIdx: SV_GroupThreadID, uint3 blockIdx: SV_GroupID, uint3 dispatchIdx: SV_DispatchThreadID) {
  // [thread_extent] blockIdx.x = 32
  // [thread_extent] threadIdx.x = 32
  for (int vthread_s = 0; vthread_s < 128; ++vthread_s) {
    output0[((((((int)blockIdx.x) * 16384) + (vthread_s * 128)) + (((int)threadIdx.x) * 4)))] = (input0[((((((int)blockIdx.x) * 16384) + (vthread_s * 128)) + (((int)threadIdx.x) * 4)))] + input1[((((((int)blockIdx.x) * 16384) + (vthread_s * 128)) + (((int)threadIdx.x) * 4)))]);
    output0[(((((((int)blockIdx.x) * 16384) + (vthread_s * 128)) + (((int)threadIdx.x) * 4)) + 1))] = (input0[(((((((int)blockIdx.x) * 16384) + (vthread_s * 128)) + (((int)threadIdx.x) * 4)) + 1))] + input1[(((((((int)blockIdx.x) * 16384) + (vthread_s * 128)) + (((int)threadIdx.x) * 4)) + 1))]);
    output0[(((((((int)blockIdx.x) * 16384) + (vthread_s * 128)) + (((int)threadIdx.x) * 4)) + 2))] = (input0[(((((((int)blockIdx.x) * 16384) + (vthread_s * 128)) + (((int)threadIdx.x) * 4)) + 2))] + input1[(((((((int)blockIdx.x) * 16384) + (vthread_s * 128)) + (((int)threadIdx.x) * 4)) + 2))]);
    output0[(((((((int)blockIdx.x) * 16384) + (vthread_s * 128)) + (((int)threadIdx.x) * 4)) + 3))] = (input0[(((((((int)blockIdx.x) * 16384) + (vthread_s * 128)) + (((int)threadIdx.x) * 4)) + 3))] + input1[(((((((int)blockIdx.x) * 16384) + (vthread_s * 128)) + (((int)threadIdx.x) * 4)) + 3))]);
  }
}";
            // Load and Compile shaders
            var hShader = dxShaderLoad(antares_code);
            if (hShader == IntPtr.Zero)
                throw new Exception("Invalid Shader Source for Compilation.");

            // Allocate device memory for inputs and outputs
            var d_input0 = dxMemAlloc(524288 * sizeof(float) * 3);
            var d_input1 = IntPtr.Add(d_input0, 524288 * sizeof(float));
            var d_output0 = IntPtr.Add(d_input1, 524288 * sizeof(float));

            var h_input0 = new float[524288];
            var h_input1 = new float[524288];
            var h_output0 = new float[524288];

            // Initialize input data in host memory
            for (int x = 0; x < h_input0.Length; ++x)
              h_input0[x] = 1;
            for (int x = 0; x < h_input1.Length; ++x)
              h_input1[x] = 2;

            // Initialize input data in device memory
            dxMemcpyHtoDAsync(d_input0, Marshal.UnsafeAddrOfPinnedArrayElement(h_input0, 0), 524288 * sizeof(float), IntPtr.Zero);
            dxMemcpyHtoDAsync(d_input1, Marshal.UnsafeAddrOfPinnedArrayElement(h_input1, 0), 524288 * sizeof(float), IntPtr.Zero);

            // Execute the shader with device memory arguments
            dxShaderLaunchAsync(hShader, new IntPtr[]{d_input0, d_input1, d_output0}, IntPtr.Zero);

            // Copy result from device memory to host memory
            dxMemcpyDtoHAsync(Marshal.UnsafeAddrOfPinnedArrayElement(h_output0, 0), d_output0, 524288 * sizeof(float), IntPtr.Zero);
            // Wait for all to complete
            dxStreamSynchronize(IntPtr.Zero);

            // Print the result in host memory
            // 1 + 2 == 3
            Console.WriteLine("Result = [" + h_output0[0] + ", " + h_output0[1] + ", .., " + h_output0[h_output0.Length - 1] + "]");

            UnloadHlslImportedDll();
            return 0;
        }
    }
}
