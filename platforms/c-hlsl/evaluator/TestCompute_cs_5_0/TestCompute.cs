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
        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dxStreamCreate();
        
        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxStreamSubmit(IntPtr hStream);
        
        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxStreamDestroy(IntPtr hStream);

        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxStreamSynchronize(IntPtr hStream);

        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dxShaderLoad(string source, [Optional] int num_outputs, [Optional] int num_inputs);
        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dxShaderLoad(string source, out int num_outputs, out int num_inputs);

        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dxShaderUnload(IntPtr hShader);

        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxShaderGetProperty(IntPtr hShader, int arg_index, out long num_elements, out long type_size, out IntPtr dtype_name);

        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxShaderLaunchAsync(IntPtr hShader, IntPtr[] source, IntPtr hStream);

        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dxEventCreate();
        
        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxEventDestroy(IntPtr hEvent);
        
        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxEventRecord(IntPtr hEvent, IntPtr hStream);
        
        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern float dxEventElapsedTime(IntPtr hStart, IntPtr hStop);

        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dxMemAlloc(long bytes);
        
        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxMemFree(IntPtr dptr);

        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxMemcpyHtoDAsync(IntPtr dptr, IntPtr hptr, long bytes, IntPtr hStream);

        [DllImport(@"antares_hlsl_v0.1_x64.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int dxMemcpyDtoHAsync(IntPtr hptr, IntPtr dptr, long bytes, IntPtr hStream);

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
                Console.Error.WriteLine("[WARN] Failed to add safe TDR settings into Windows registry.");
                Console.Error.WriteLine("[WARN] Super privilege is required for safe TDR settings.");
                Console.Error.WriteLine("[WARN] Otherwrise, invalid shaders might trigger Win10's blue screen.");
                Console.Error.WriteLine();
            }
        }

        static void Main(string[] args)
        {
            initRegistryForSafeTDR();
            var shader_file = @".\dx_kernel.hlsl";
            if (!File.Exists(shader_file))
                throw new Exception("No Shader Source Found to Compile.");

            // HLSL source code generated by Antares HLSL backend
            var antares_code = "file://" + shader_file;
            int num_inputs = 0, num_outputs = 0;
            var hShader = dxShaderLoad(antares_code, out num_inputs, out num_outputs);
            if (hShader == IntPtr.Zero)
                throw new Exception("Invalid Shader Source for Compilation.");

            // saving properties from shader program
            var num_elements = new long[num_inputs + num_outputs];
            var type_size = new long[num_inputs + num_outputs];
            var dtype_name = new string[num_inputs + num_outputs];
            var kargs = new IntPtr[num_inputs + num_outputs];

            // create device buffers for inputs and outputs
            for (int i = 0; i < num_elements.Length; ++i)
            {
                IntPtr dtype_ptr = IntPtr.Zero;
                dxShaderGetProperty(hShader, i, out num_elements[i], out type_size[i], out dtype_ptr);
                dtype_name[i] = Marshal.PtrToStringAnsi(dtype_ptr);
                
                if (i < num_inputs)
                    Console.WriteLine("InputArg " + i + ": NumElements = " + num_elements[i] + ", TypeBytes = " + type_size[i] + ", TypeName = " + dtype_name[i]);
                else
                    Console.WriteLine("OutputArg " + (i - num_inputs) + ": NumElements = " + num_elements[i] + ", TypeBytes = " + type_size[i] + ", TypeName = " + dtype_name[i]);

                kargs[i] = dxMemAlloc(num_elements[i] * type_size[i]);
                Debug.Assert(kargs[i] != IntPtr.Zero);
            }

            // fill input data
            for (int i = 0; i < num_inputs; ++i)
            {
                if (dtype_name[i] == "int32")
                {
                    var h_input = new int[num_elements[i]];
                    for (int x = 0; x < h_input.Length; ++x)
                        h_input[x] = (x + i + 1) % 71;
                    dxMemcpyHtoDAsync(kargs[i], Marshal.UnsafeAddrOfPinnedArrayElement(h_input, 0), num_elements[i] * type_size[i], IntPtr.Zero);
                }
                else
                {
                    Debug.Assert((num_elements[i] * type_size[i]) % sizeof(float) == 0);
                    var h_input = new float[(num_elements[i] * type_size[i]) / sizeof(float)];
                    for (int x = 0; x < h_input.Length; ++x)
                        h_input[x] = (x + i + 1) % 71;
                    dxMemcpyHtoDAsync(kargs[i], Marshal.UnsafeAddrOfPinnedArrayElement(h_input, 0), num_elements[i] * type_size[i], IntPtr.Zero);
                }
            }

            // compute results in background
            dxShaderLaunchAsync(hShader, kargs, IntPtr.Zero);

            // read results back and compute digest
            for (int i = num_inputs; i < num_elements.Length; ++i)
            {
                int output_id = i - num_inputs;
                double digest = 0.0;

                if (dtype_name[i] == "int32")
                {
                    var h_output = new int[num_elements[i]];
                    dxMemcpyDtoHAsync(Marshal.UnsafeAddrOfPinnedArrayElement(h_output, 0), kargs[i], num_elements[i] * type_size[i], IntPtr.Zero);
                    dxStreamSynchronize(IntPtr.Zero);
                    for (int x = 0; x < h_output.Length; ++x)
                        digest += (x + 1) % 83 * h_output[x];
                }
                else
                {
                    Debug.Assert((num_elements[i] * type_size[i]) % sizeof(float) == 0);
                    var h_output = new float[(num_elements[i] * type_size[i]) / sizeof(float)];
                    dxMemcpyDtoHAsync(Marshal.UnsafeAddrOfPinnedArrayElement(h_output, 0), kargs[i], num_elements[i] * type_size[i], IntPtr.Zero);
                    dxStreamSynchronize(IntPtr.Zero);
                    for (int x = 0; x < h_output.Length; ++x)
                        digest += (x + 1) % 83 * h_output[x];
                }

                Console.WriteLine("- K/" + output_id + " = " + digest);
            }

            var hStart = dxEventCreate();
            var hStop = dxEventCreate();
            dxEventRecord(hStart, IntPtr.Zero);
            dxShaderLaunchAsync(hShader, kargs, IntPtr.Zero);
            dxEventRecord(hStop, IntPtr.Zero);
            dxStreamSynchronize(IntPtr.Zero);

            float time_in_sec = dxEventElapsedTime(hStart, hStop);
            var num_runs = Math.Max(3, Math.Min(10000, Convert.ToInt32(1.0 / time_in_sec)));

            dxEventRecord(hStart, IntPtr.Zero);
            for (int i = 0; i < num_runs; ++i)
                dxShaderLaunchAsync(hShader, kargs, IntPtr.Zero);
            dxEventRecord(hStop, IntPtr.Zero);
            dxStreamSynchronize(IntPtr.Zero);

            float tpr = dxEventElapsedTime(hStart, hStop) / num_runs;
            Console.WriteLine("- TPR = " + tpr);
        }
    }
}
