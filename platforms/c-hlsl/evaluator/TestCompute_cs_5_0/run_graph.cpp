// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#if 1

#include <windows.h>
#include <cstdio>
#include <vector>
#include <string>
#include <chrono>

#define FATAL_ERROR(msg) \
    fprintf(stderr, "[Error] %s", msg), _exit(1);

int main(int argc, char** argv)
{
    HINSTANCE hAntaresHlslLib = LoadLibrary(L"antares_hlsl_x64_v0.1.dll");
    if (hAntaresHlslLib == nullptr)
        FATAL_ERROR("Cannot load antares hlsl DLL library.");

    auto dxLoadShader = (void* (*)(const char*, int*, int*))::GetProcAddress(hAntaresHlslLib, "dxLoadShader");
    auto dxAllocateBuffer = (void* (*)(size_t))::GetProcAddress(hAntaresHlslLib, "dxAllocateBuffer");
    auto dxGetShaderArgumentProperty = (void (*)(void*, int, size_t*, size_t*, const char**))::GetProcAddress(hAntaresHlslLib, "dxGetShaderArgumentProperty");
    auto dxMemcpyHostToBuffer = (void (*)(void*, void*, size_t))::GetProcAddress(hAntaresHlslLib, "dxMemcpyHostToBuffer");
    auto dxMemcpyBufferToHost = (void (*)(void*, void*, size_t))::GetProcAddress(hAntaresHlslLib, "dxMemcpyBufferToHost");
    auto dxLaunchShader = (void (*)(void*, void**))::GetProcAddress(hAntaresHlslLib, "dxLaunchShader");
    auto dxSynchronize = (void (*)())::GetProcAddress(hAntaresHlslLib, "dxSynchronize");

    int num_inputs, num_outputs;
    auto handle = dxLoadShader("file://dx_kernel.hlsl", &num_inputs, &num_outputs);
    if (handle == nullptr)
        FATAL_ERROR("Invalid hlsl source code in complation.");

    std::vector<void*> kargs(size_t(num_inputs) + size_t(num_outputs));
    std::vector<size_t> counts(kargs.size());
    std::vector<std::string> dtypes(kargs.size());

    for (int i = 0; i < kargs.size(); ++i) {
        const char* dtype_ptr;
        size_t type_size;
        dxGetShaderArgumentProperty(handle, i, &counts[i], &type_size, &dtype_ptr);
        dtypes[i] = dtype_ptr;
        kargs[i] = dxAllocateBuffer(counts[i] * type_size);

        if (i < num_inputs)
        {
            if (dtypes[i] == "float32")
            {
                std::vector<float> h_input(counts[i]);
                for (int k = 0; k < h_input.size(); ++k)
                    h_input[k] = (i + k + 1) % 71;
                dxMemcpyHostToBuffer(kargs[i], h_input.data(), h_input.size() * sizeof(h_input[0]));
                dxSynchronize();
            }
            else if (dtypes[i] == "int32")
            {
                std::vector<int> h_input(counts[i]);
                for (int k = 0; k < h_input.size(); ++k)
                    h_input[k] = (i + k + 1) % 71;
                dxMemcpyHostToBuffer(kargs[i], h_input.data(), h_input.size() * sizeof(h_input[0]));
                dxSynchronize();
            }
            else
                FATAL_ERROR(("Unhandled value initialization for dtype name: " + dtypes[i]).c_str());

            printf("InputArg %d: NumElements = %zd, TypeBytes = %zd, TypeName = %s\n", i, counts[i], type_size, dtypes[i].c_str());
        }
        else
        {
            printf("OutputArg %d: NumElements = %zd, TypeBytes = %zd, TypeName = %s\n", i - num_inputs, counts[i], type_size, dtypes[i].c_str());
        }
    }
    printf("\n");

    dxLaunchShader(handle, kargs.data());
    if (num_outputs <= 0)
        FATAL_ERROR("No output result for evaluation.");

    for (int i = num_inputs; i < kargs.size(); ++i)
    {
        double digest = 0.0;
        int output_id = i - num_inputs;
        if (dtypes[output_id] == "float32")
        {
            std::vector<float> h_output(counts[i]);
            dxMemcpyBufferToHost(h_output.data(), kargs[i], h_output.size() * sizeof(h_output[0]));
            dxSynchronize();
            for (int k = 0; k < h_output.size(); ++k)
                digest += double((k + output_id + 1) % 83) * h_output[k];
        }
        else if (dtypes[output_id] == "int32")
        {
            std::vector<int> h_output(counts[i]);
            dxMemcpyBufferToHost(h_output.data(), kargs[i], h_output.size() * sizeof(h_output[0]));
            dxSynchronize();
            for (int k = 0; k < h_output.size(); ++k)
                digest += double((k + output_id + 1) % 83) * h_output[k];
        }
        else
            FATAL_ERROR(("Unhandled result digest calculation for dtype name: " + dtypes[i]).c_str());
        printf("- K/%d = %.6e\n", output_id, digest);
    }

    auto start = std::chrono::high_resolution_clock::now();
    dxLaunchShader(handle, kargs.data());
    dxSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();

    double elasped_sec = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

    int num_runs = int(1 / (elasped_sec > 1e-3 ? elasped_sec : 1e-3));
    num_runs = num_runs > 3 ? num_runs : 3;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i)
    {
        dxLaunchShader(handle, kargs.data());
    }
    dxSynchronize();
    stop = std::chrono::high_resolution_clock::now();

    elasped_sec = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() / num_runs;
    printf("- TPR = %.6e\n", elasped_sec);
    return 0;
}

#else

#include "d3dx12_antares.h"

int main(int argc, char** argv)
{
    using namespace antares;

    D3DDevice device(false, false);
    device.Init();
    
#ifdef _USE_GPU_TIMER_
    device.LockGPUClock();
#endif

    std::vector<ID3D12CommandList*> cmdQueue;

    {
        std::ifstream t("dx_kernel.hlsl", ios_base::binary);
        if (t.fail())
            exit(1);
        std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        int pos = str.find("///") + sizeof("///") - 1;
        assert(pos >= sizeof("///") - 1);
        int sep = str.find(':', pos);
        assert(sep >= 0);
        int tail = str.find('\n', sep);
        assert(tail >= 0);
        std::string s_in = str.substr(pos, sep - pos) + ",", s_out = str.substr(sep + 1, tail - sep - 1) + ",";

        struct TensorProp {
            std::string name, dtype;
            int bits, lanes;
            std::vector<size_t> shape;

            size_t count() const {
                size_t ret = 1;
                for (auto it : shape)
                    ret *= it;
                return ret;
            }
        };

        std::vector<NNfusionTensor> input_buffers, output_buffers;
        std::vector<TensorProp> inputs, outputs;
        std::vector<NNfusionMemcpy> memcpy_cmds, results;
        std::vector<size_t> grids;

        auto parse_shape = [](TensorProp &tensor, const std::string &shape) -> void {
            tensor.shape.clear();
            for (int i = 0, j = 1; j <= shape.size(); ++j)
                if (j == shape.size() || shape[j] == '-')
                    tensor.shape.push_back(std::atoi(shape.substr(i, j - i).c_str())), i = j + 1;
        };

        auto parse_dtype = [](TensorProp& tensor, const std::string& dtype) -> void {
            int idx = dtype.find(']');
            if (idx >= 0) {
                int at = dtype.find('[') + 1;
                tensor.dtype = dtype.substr(at, idx - at);
                at = dtype.find('x', idx);
                if (at < 0) {
                    tensor.bits = std::atoi(dtype.substr(idx + 1).c_str());
                    tensor.lanes = 1;
                }
                else {
                    tensor.bits = std::atoi(dtype.substr(idx + 1, at - idx - 1).c_str());
                    tensor.lanes = std::atoi(dtype.substr(at + 1).c_str());
                }
            }
            else {
                for (idx = dtype.size() - 1; idx >= 0; --idx)
                    if (!isdigit(dtype[idx]))
                        break;
                if (idx >= dtype.size())
                    tensor.bits = 32;
                else
                    tensor.bits = std::atoi(dtype.substr(idx + 1).c_str());
                tensor.lanes = 1;
                tensor.dtype = dtype;
            }
        };

        for (pos = 0; sep = s_in.find(',', pos), sep >= 0; pos = sep + 1) {
            auto prop = s_in.substr(pos, size_t(sep - pos));
            int left = prop.find('/'), right = prop.find('/', left + 1L);

            TensorProp tensor_prop;
            parse_shape(tensor_prop, prop.substr(0, left));
            parse_dtype(tensor_prop, prop.substr(left + 1L, right - left - 1L));
            tensor_prop.name = prop.substr(right + 1);
            printf("+ Input: name = %s, dtype = %s, bits = %d, lanes = %d, count = %zd\n", tensor_prop.name.c_str(), tensor_prop.dtype.c_str(), tensor_prop.bits, tensor_prop.lanes, tensor_prop.count());

            input_buffers.emplace_back(NNfusionTensor(device, { tensor_prop.shape }, tensor_prop.bits * tensor_prop.lanes / 8));

            int i = input_buffers.size() - 1;
            if (tensor_prop.dtype == "int32")
                memcpy_cmds.emplace_back(device, cmdQueue, input_buffers[i], load_data<int>(i, tensor_prop.count() * tensor_prop.lanes));
            else if (tensor_prop.dtype == "float32")
                memcpy_cmds.emplace_back(device, cmdQueue, input_buffers[i], load_data<float>(i, tensor_prop.count() * tensor_prop.lanes));
            else
                memcpy_cmds.emplace_back(device, cmdQueue, input_buffers[i], load_data<char>(i, tensor_prop.count() * tensor_prop.lanes * tensor_prop.bits / 8));
            inputs.push_back(std::move(tensor_prop));
        }
        for (pos = 0; sep = s_out.find(',', pos), sep >= 0; pos = sep + 1) {
            auto prop = s_out.substr(pos, size_t(sep - pos));
            int left = prop.find('/'), right = prop.find('/', left + 1L);

            TensorProp tensor_prop;
            parse_shape(tensor_prop, prop.substr(0, left));
            parse_dtype(tensor_prop, prop.substr(left + 1L, right - left - 1L));
            tensor_prop.name = prop.substr(right + 1);
            printf("+ Output: name = %s, dtype = %s, bits = %d, lanes = %d, count = %zd\n", tensor_prop.name.c_str(), tensor_prop.dtype.c_str(), tensor_prop.bits, tensor_prop.lanes, tensor_prop.count());

            output_buffers.emplace_back(NNfusionTensor(device, { tensor_prop.shape }, tensor_prop.bits* tensor_prop.lanes / 8));
            outputs.push_back(std::move(tensor_prop));
        }

        int at_x = str.find("blockIdx.x = "), bx = (at_x >= 0) ? std::atoi(str.c_str() + at_x + sizeof("blockIdx.x = ") - 1): 1;
        int at_y = str.find("blockIdx.y = "), by = (at_y >= 0) ? std::atoi(str.c_str() + at_y + sizeof("blockIdx.y = ") - 1) : 1;
        int at_z = str.find("blockIdx.z = "), bz = (at_z >= 0) ? std::atoi(str.c_str() + at_z + sizeof("blockIdx.z = ") - 1) : 1;

        grids.push_back(bx);
        grids.push_back(by);
        grids.push_back(bz);

        assert(inputs.size() >= 0 && outputs.size() > 0);
        assert(grids.size() == 3);
        printf("+ Groups = (%u, %u, %u)\n", bx, by, bz);

        NNfusionOperator op_Compute_Kernel(device, cmdQueue, input_buffers, output_buffers, grids, str.c_str());
        pos = cmdQueue.size() - 1;

        for (int i = 0; i < outputs.size(); ++i) {
            results.emplace_back(device, cmdQueue, nullptr, output_buffers[i]);
        }

        std::chrono::high_resolution_clock::time_point t1, t2;
        double sec;

        t1 = std::chrono::high_resolution_clock::now();
        printf(">> Evalute DxCompute ..\n\n");
        for (auto cmdList : cmdQueue)
        {
            device.pCommandQueue->ExecuteCommandLists(1, &cmdList);
        }
        device.AwaitExecution();
        for (int i = 0; i < results.size(); ++i) {
            double ans = results[i].PrintStageBuffer<float>(device);
            printf("- K/%d = %.10e\n", i, ans);
        }
        t2 = std::chrono::high_resolution_clock::now();
        sec = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        int NUM_STEPS;
        if (sec > 1)
            NUM_STEPS = 1;
        else if (sec > 0.1)
            NUM_STEPS = 3;
        else if (sec > 0.01)
            NUM_STEPS = 10;
        else
            NUM_STEPS = 100;
       
#ifdef _USE_GPU_TIMER_
        sec = 0;
        for (int i = 0; i < NUM_STEPS; i++) {
            device.pCommandQueue->ExecuteCommandLists(1, cmdQueue.data() + pos);
            device.AwaitExecution();
            device.SyncTimerData();
            sec += device.GetTime(op_Compute_Kernel.TimerIndex());
        }
        sec /= (double)NUM_STEPS;
#else
        t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_STEPS; i++) {
            device.pCommandQueue->ExecuteCommandLists(1, cmdQueue.data() + pos);
        }
        device.AwaitExecution();
        t2 = std::chrono::high_resolution_clock::now();
        sec = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() / NUM_STEPS;
#endif
        printf("- TPR = %.6e\n", sec);
    }
    return 0;
}
#endif
