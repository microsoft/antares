// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// This program is for `Win10-x64bit` only.
// [WSL CS Compiler] Using Command: C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe /out:program.exe program.cs

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Threading;

using Microsoft.Win32;
using System.Diagnostics;

namespace AntaresHlslEvalAgent
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

        static bool isSafeTDRConfigured()
        {
            try
            {
                using (RegistryKey key = Registry.CurrentUser.CreateSubKey(@"SOFTWARE\Antares\HLSL"))
                {
                    var value = key.GetValue("TdrConfigState");
                    key.Close();
                    if (value == null || Convert.ToInt32(value) != 3)
                        return false;
                    return true;
                }
            }
            catch (UnauthorizedAccessException)
            {
                return false;
            }
        }

        static string runSystemCommand(string program, string args, int timeout_in_ms = -1)
        {
            ProcessStartInfo processInfo = new ProcessStartInfo();
            processInfo.FileName = program;
            processInfo.Arguments = args;
            processInfo.RedirectStandardError = true;
            processInfo.RedirectStandardOutput = true;
            processInfo.UseShellExecute = false;
            processInfo.CreateNoWindow = true;

            Process process = new Process();
            process.StartInfo = processInfo;
            process.Start();

            string output = "";
            if (!process.WaitForExit(timeout_in_ms))
                process.Kill();
            else
                output = process.StandardOutput.ReadToEnd();
            process.Close();
            return output;
        }

        public const int LISTEN_PORT = 8860;

        static bool initEnvironment()
        {
            if (!isSafeTDRConfigured())
            {
                string tdr_path = AppDomain.CurrentDomain.BaseDirectory;
                if (!tdr_path.EndsWith(@"\"))
                    tdr_path += @"\";
                tdr_path += "TDR.reg";

                Console.WriteLine("[INFO] Safe TDR settings is not configured. Trying to configure this session by registry file:\n\t" + tdr_path);
                runSystemCommand(@"powershell.exe", @"Start-Process -Verb runas -FilePath regedit.exe -ArgumentList '/s', '" + tdr_path + "'");

                Console.WriteLine();
                Thread.Sleep(2000);

                if (!isSafeTDRConfigured())
                {
                    Console.WriteLine("[WARN] No access to initialize safe TDR settings from Windows registry.");
                    Console.WriteLine("[WARN] Super privilege is recommended for the first run of this program.");
                    Console.WriteLine("[WARN] Otherwrise, invalid shader execution might trigger Win10's blue screen.");
                    Console.WriteLine();
                    return false;
                }
                else
                {
                    Console.WriteLine("[INFO] Successfully adding configuration for safe TDR.");
                    Console.WriteLine();
                }
            }
            if (!File.Exists(@".\antares_hlsl_v0.1_x64.dll"))
            {
                Console.WriteLine("[INFO] Downloading required DLL dependencies..");
                runSystemCommand("curl.exe", "-LOs https://github.com/microsoft/antares/raw/library/antares_hlsl_v0.1_x64.dll");
            }

            runSystemCommand("netsh", "advfirewall firewall add rule name=\"TCP Port for Antares\" dir=in action=allow protocol=TCP localport=" + LISTEN_PORT);
            return true;
        }

        static string GetBetween(string source, string begin, string end)
        {
            int start = source.IndexOf(begin);
            if (start < 0)
                return "";
            start += begin.Length;
            int stop = source.IndexOf(end, start);
            if (stop < 0)
                return "";
            return source.Substring(start, stop - start);
        }

        static async Task HandleIncomingConnections(HttpListener listener)
        {
            bool runServer = true;
            while (runServer)
            {
                HttpListenerContext ctx = await listener.GetContextAsync();
                HttpListenerRequest req = ctx.Request;
                HttpListenerResponse resp = ctx.Response;

                Console.WriteLine("[INFO] ============================================");
                Console.WriteLine("[INFO] Receive a request from " + req.RemoteEndPoint + " through HTTP " + req.HttpMethod);

                if (req.HttpMethod == "PUT" && req.HasEntityBody)
                {
                    using (System.IO.Stream body = req.InputStream)
                    {
                        using (System.IO.StreamReader reader = new System.IO.StreamReader(body, req.ContentEncoding))
                        {
                            string source = reader.ReadToEnd();
                            System.IO.File.WriteAllText(@".\dx_kernel.hlsl", source);
                            string expected_timeout = req.Headers.Get("ET").Trim();

                            string result = runSystemCommand(@".\" + System.Diagnostics.Process.GetCurrentProcess().ProcessName, expected_timeout, 5000);

                            byte[] buffer = System.Text.Encoding.UTF8.GetBytes(result);
                            resp.OutputStream.Write(buffer, 0, buffer.Length);
                            resp.OutputStream.Close();
                        }
                    }
                }
                resp.Close();
            }
        }

        static int loopServerForever()
        {
            Console.WriteLine("[INFO] Antares HLSL Evaluator Agent is listening on TCP port :" + LISTEN_PORT);
            Console.WriteLine("[INFO] Possible valid AGENT_URL for Antares HLSL client includes:");
            Console.WriteLine(runSystemCommand("cmd.exe", "/c \"ipconfig | findstr IPv4\""));

            var listener = new HttpListener();
            listener.Prefixes.Add("http://*:" + LISTEN_PORT + "/");
            listener.Start();

            Task listenTask = HandleIncomingConnections(listener);
            listenTask.GetAwaiter().GetResult();
            listener.Close();
            return 0;
        }

        static int Main(string[] args)
        {
            bool configured = initEnvironment();
            float expected_timeout = -1.0f;

            if (args.Length >= 1)
            {
                if (args[0] == "server")
                {
                    Console.WriteLine("[INFO] Config initialization finished: status = " + configured);
                    return loopServerForever();
                }
                else
                    expected_timeout = (float)Double.Parse(args[0]);
            }

            var shader_file = @".\dx_kernel.hlsl";
            if (!File.Exists(shader_file))
            {
                Console.WriteLine("No Shader Source Found to Evaluate.");
                return 1;
            }

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

            Console.Write("\n- {");
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
                Console.Write("\"K/" + output_id + "\": " + String.Format("{0:E10}", digest) + ", ");
            }

            var hStart = dxEventCreate();
            var hStop = dxEventCreate();
            dxEventRecord(hStart, IntPtr.Zero);
            dxShaderLaunchAsync(hShader, kargs, IntPtr.Zero);
            dxEventRecord(hStop, IntPtr.Zero);
            dxStreamSynchronize(IntPtr.Zero);

            float time_in_sec = dxEventElapsedTime(hStart, hStop);
            var num_runs = Math.Max(3, Math.Min(10000, Convert.ToInt32(1.0 / time_in_sec)));

            if (expected_timeout > 0 && time_in_sec >= expected_timeout)
                num_runs = 1;


            dxEventRecord(hStart, IntPtr.Zero);
            for (int i = 0; i < num_runs; ++i)
                dxShaderLaunchAsync(hShader, kargs, IntPtr.Zero);
            dxEventRecord(hStop, IntPtr.Zero);
            dxStreamSynchronize(IntPtr.Zero);

            float tpr = dxEventElapsedTime(hStart, hStop) / num_runs;
            Console.WriteLine("\"TPR\": " + tpr + "}");
            return 0;
        }
    }
}
