// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


#include "D3D12APIWrapper.h"

#ifndef _API_WRAPPER_V2_
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string>
#include <cassert>
#include <unordered_map>
#include <vector>

#include "d3dx12_antares.h"

#define __EXPORT__ extern "C" __declspec(dllexport)

struct dx_buffer_t {
    size_t size;
    ComPtr<ID3D12Resource> handle;
};

struct dx_tensor_t {
    std::vector<size_t> shape;
    std::string name, dtype;

    size_t NumElements() {
        return std::accumulate(shape.begin(), shape.end(), 1L, std::multiplies<size_t>());
    }

    size_t TypeSize() {
        for (int i = dtype.size() - 1; i >= 0; --i) {
            if (!isdigit(dtype[i])) {
                int bits = std::atoi(dtype.c_str() + i + 1);
                if (bits % 8 > 0)
                    throw std::runtime_error("Data type bitsize must align with 8-bit byte type.");
                return bits / 8;
            }
        }
        throw std::runtime_error(("Invalid data type name: " + dtype).c_str());
    }
};

struct dx_shader_t {
    int block[3], thread[3];
    std::vector<dx_tensor_t> inputs, outputs;
    std::string source;
    CD3DX12_SHADER_BYTECODE bytecode;
#ifdef _USE_DXC_
    ComPtr<IDxcBlob> computeShader;
#else
    ComPtr<ID3DBlob> computeShader;
#endif
};

static antares::D3DDevice device(false, false);
static std::unordered_map<size_t, std::vector<void*>> bufferDict;

static std::string get_between(const std::string& source, const std::string& begin, const std::string& end, const char* def = "") {
    std::string ret;
    int idx = source.find(begin);
    if (idx < 0)
        return def;
    idx += begin.size();
    int tail = source.find(end, idx);
    if (idx < 0)
        return def;
    return source.substr(idx, tail - idx);
}

__EXPORT__ int dxInit(int flags)
{
    static bool inited = false;
    if (!inited)
    {
        inited = true;
        device.Init();
    }
    return 0;
}

__EXPORT__ void* dxAllocateBuffer(size_t bytes)
{
    if (dxInit(0) != 0)
        return nullptr;

    auto buffs = bufferDict[bytes];
    if (buffs.size() > 0)
    {
        void* ret = buffs.back();
        buffs.pop_back();
        return ret;
    }
    auto* buff = new dx_buffer_t;
    buff->size = bytes;
    device.CreateGPUOnlyResource(bytes, &buff->handle);
    assert(buff->handle.Get() != nullptr);
    return buff;
}

__EXPORT__ void dxReleaseBuffer(void* dptr)
{
    auto _buff = (dx_buffer_t*)(dptr);
    bufferDict[_buff->size].push_back(dptr);
}

__EXPORT__ void dxGetShaderArgumentProperty(void* handle, int arg_index, size_t* num_elements, size_t* type_size, const char** dtype_name)
{
    auto hd = (dx_shader_t*)handle;
    size_t count, tsize;
    std::string dtype;
    if (arg_index < hd->inputs.size())
    {
        count = hd->inputs[arg_index].NumElements();
        dtype = hd->inputs[arg_index].dtype;
        tsize = hd->inputs[arg_index].TypeSize();
    }
    else
    {
        count = hd->outputs[arg_index - hd->inputs.size()].NumElements();
        dtype = hd->outputs[arg_index - hd->inputs.size()].dtype;
        tsize = hd->outputs[arg_index - hd->inputs.size()].TypeSize();
    }
    if (num_elements != nullptr)
        *num_elements = count;
    if (type_size != nullptr)
        *type_size = tsize;
    if (dtype_name != nullptr)
    {
        static char lastDtypeName[MAX_PATH];
        strncpy_s(lastDtypeName, dtype.c_str(), sizeof(lastDtypeName));
        *dtype_name = lastDtypeName;
    }
}

__EXPORT__ void* dxLoadShader(const char* _source, int *num_inputs, int *num_outputs)
{
    if (dxInit(0) != 0)
        return nullptr;

    std::string source = _source;
    const char proto[] = "file://";
    if (strncmp(source.c_str(), proto, sizeof(proto) - 1) == 0) {
        std::ifstream t(_source + sizeof(proto) - 1, ios_base::binary);
        if (t.fail())
            return nullptr;
        std::string _((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        source = std::move(_);
    }

    auto* handle = new dx_shader_t;
    handle->source = source;

#ifdef _USE_DXC_
    // Use cs_6_0 since dxc only supports cs_6_0 or higher shader models.
    //IDxcBlob* computeShader = nullptr;
    //computeShader = antares::DXCompiler::Get()->Compile(source.data(), source.size(), L"CSMain", L"cs_6_0").Get(); // Bug fix.
    auto computeShader = antares::DXCompiler::Get()->Compile(source.data(), source.size(), L"CSMain", L"cs_6_0");
    if (computeShader != nullptr)
        handle->bytecode = CD3DX12_SHADER_BYTECODE(computeShader->GetBufferPointer(), computeShader->GetBufferSize());
#else
    ComPtr<ID3DBlob> computeShader = nullptr;
    if (D3DCompile(source.data(), source.size(), NULL, NULL, NULL, "CSMain", "cs_5_1", 0, 0, &computeShader, NULL) >= 0 && computeShader != nullptr)
        handle->bytecode = CD3DX12_SHADER_BYTECODE(computeShader.Get());
#endif
    if (computeShader == nullptr) {
        delete handle;
        return nullptr;
    }
    handle->computeShader = computeShader;

    auto ssplit = [](const std::string& source, const std::string& delim) -> std::vector<std::string> {
        std::vector<std::string> ret;
        int it = 0, next;
        while (next = source.find(delim, it), next >= 0) {
            ret.push_back(source.substr(it, next - it));
            it = next + delim.size();
        }
        ret.push_back(source.substr(it));
        return std::move(ret);
    };

    auto parse_tensor = [&](const std::string& param) -> dx_tensor_t {
        dx_tensor_t ret;
        auto parts = ssplit(param, "/");
        for (auto it : ssplit(parts[0], "-"))
            ret.shape.push_back(std::atoi(it.c_str()));
        ret.dtype = parts[1];
        ret.name = parts[2];
        return ret;
    };

    auto str_params = get_between(source, "///", "\n");
    auto arr_params = ssplit(str_params, ":");
    assert(arr_params.size() == 2);
    auto in_params = ssplit(arr_params[0], ","), out_params = ssplit(arr_params[1], ",");

    for (int i = 0; i < in_params.size(); ++i)
        handle->inputs.push_back(parse_tensor(in_params[i]));
    for (int i = 0; i < out_params.size(); ++i)
        handle->outputs.push_back(parse_tensor(out_params[i]));

    handle->block[0] = std::atoi(get_between(source, "// [thread_extent] blockIdx.x = ", "\n", "1").c_str());
    handle->block[1] = std::atoi(get_between(source, "// [thread_extent] blockIdx.y = ", "\n", "1").c_str());
    handle->block[2] = std::atoi(get_between(source, "// [thread_extent] blockIdx.z = ", "\n", "1").c_str());
    handle->thread[0] = std::atoi(get_between(source, "// [thread_extent] threadIdx.x = ", "\n", "1").c_str());
    handle->thread[1] = std::atoi(get_between(source, "// [thread_extent] threadIdx.y = ", "\n", "1").c_str());
    handle->thread[2] = std::atoi(get_between(source, "// [thread_extent] threadIdx.z = ", "\n", "1").c_str());

    assert(INT64(handle->thread[0]) * handle->thread[1] * handle->thread[2] <= 1024);
    if (num_inputs != nullptr)
        *num_inputs = handle->inputs.size();
    if (num_outputs != nullptr)
        *num_outputs = handle->outputs.size();
    return handle;
}

struct LaunchShaderAsyncResource {
    ComPtr<ID3D12GraphicsCommandList> m_computeCommandList;
    ComPtr<ID3D12CommandAllocator> m_computeCommandAllocator;

    // Compute Command Only
    ComPtr<ID3D12RootSignature> m_computeRootSignature;
    ComPtr<ID3D12PipelineState> m_computeState;

    // Memcpy Command Only
    ComPtr<ID3D12Resource> deviceCPUSrcX;
    void *src, *dst;
    size_t bytes;

    LaunchShaderAsyncResource(
        ComPtr<ID3D12GraphicsCommandList> m_computeCommandList,
        ComPtr<ID3D12CommandAllocator> m_computeCommandAllocator,
        ComPtr<ID3D12RootSignature> m_computeRootSignature,
        ComPtr<ID3D12PipelineState> m_computeState) :
        m_computeCommandList(m_computeCommandList),
        m_computeCommandAllocator(m_computeCommandAllocator),
        m_computeRootSignature(m_computeRootSignature),
        m_computeState(m_computeState),
        src(nullptr), dst(nullptr), bytes(0L)
    {
    }

    LaunchShaderAsyncResource(
        ComPtr<ID3D12GraphicsCommandList> m_computeCommandList,
        ComPtr<ID3D12CommandAllocator> m_computeCommandAllocator,
        ComPtr<ID3D12Resource> deviceCPUSrcX,
        void* src, void* dst, size_t bytes) :
        m_computeCommandList(m_computeCommandList),
        m_computeCommandAllocator(m_computeCommandAllocator),
        deviceCPUSrcX(deviceCPUSrcX),
        src(src), dst(dst), bytes(bytes)
    {
    }
};

static std::vector<LaunchShaderAsyncResource> reservedResources;

__EXPORT__ void dxMemcpyHostToBuffer(void* dst, void* src, size_t bytes)
{
    ComPtr<ID3D12Resource> deviceCPUSrcX;
    device.CreateUploadBuffer(bytes, &deviceCPUSrcX);

    auto dst_buffer = (dx_buffer_t*)dst;

    ComPtr<ID3D12CommandAllocator> pCommandAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_computeCommandList;
    IFE(device.pDevice->CreateCommandAllocator(device.CommandListType, IID_PPV_ARGS(&pCommandAllocator)));
    IFE(device.pDevice->CreateCommandList(0, device.CommandListType, pCommandAllocator.Get(), nullptr, IID_PPV_ARGS(&m_computeCommandList)));
    m_computeCommandList->CopyResource(dst_buffer->handle.Get(), deviceCPUSrcX.Get());
    IFE(m_computeCommandList->Close());

    reservedResources.emplace_back(m_computeCommandList, pCommandAllocator, deviceCPUSrcX, src, nullptr, bytes);
}

__EXPORT__ void dxMemcpyBufferToHost(void* dst, void* src, size_t bytes)
{
    ComPtr<ID3D12Resource> deviceCPUSrcX;
    device.CreateReadbackBuffer(bytes, &deviceCPUSrcX);

    auto src_buffer = (dx_buffer_t*)src;

    ComPtr<ID3D12CommandAllocator> pCommandAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_computeCommandList;
    IFE(device.pDevice->CreateCommandAllocator(device.CommandListType, IID_PPV_ARGS(&pCommandAllocator)));
    IFE(device.pDevice->CreateCommandList(0, device.CommandListType, pCommandAllocator.Get(), nullptr, IID_PPV_ARGS(&m_computeCommandList)));
    m_computeCommandList->CopyResource(deviceCPUSrcX.Get(), src_buffer->handle.Get());
    IFE(m_computeCommandList->Close());

    reservedResources.emplace_back(m_computeCommandList, pCommandAllocator, deviceCPUSrcX, nullptr, dst, bytes);
}

__EXPORT__ void dxLaunchShader(void* handle, void** buffers)
{
    auto hd = (dx_shader_t*)handle;

    ComPtr<ID3D12GraphicsCommandList> m_computeCommandList;
    ComPtr<ID3D12RootSignature> m_computeRootSignature;
    ComPtr<ID3D12PipelineState> m_computeState;
    ComPtr<ID3D12CommandAllocator> m_computeCommandAllocator;
    D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc{};

#ifdef _USE_DESCRIPTOR_HEAP_

    struct DescHeap {
        ComPtr<ID3D12DescriptorHeap> heap;
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
        UINT nDescStep, offsetRecord;
    };

    static DescHeap globalDescHeap;
    if (!globalDescHeap.nDescStep) {
        auto InitDescriptorHeap = [](ID3D12Device* pDevice, D3D12_DESCRIPTOR_HEAP_TYPE type, UINT nDescriptors)
        {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            memset(&desc, 0, sizeof(desc));
            ZeroMemory(&desc, sizeof(desc));
            desc.NumDescriptors = nDescriptors;
            desc.Type = type;
            desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ComPtr<ID3D12DescriptorHeap> pDescHeap;
            IFE(pDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&pDescHeap)));

            globalDescHeap.nDescStep = pDevice->GetDescriptorHandleIncrementSize(type);
            globalDescHeap.heap = pDescHeap;
            globalDescHeap.cpuHandle = pDescHeap->GetCPUDescriptorHandleForHeapStart();
            globalDescHeap.offsetRecord = 0;
        };

        // const UINT MAX_HEAP_SIZE = (1U << 20);
        // Resource binding tier1/2 devices and some of the tier3 devices (e.g. NVIDIA Turing GPUs) DO-NOT support descriptor heap size larger than 1000000.
        const UINT MAX_HEAP_SIZE = 65536;
        InitDescriptorHeap(device.pDevice.Get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, MAX_HEAP_SIZE);
        assert(globalDescHeap.nDescStep > 0);
    }

    // Prepare Root
    std::vector<CD3DX12_ROOT_PARAMETER1> computeRootParameters(1);
    CD3DX12_DESCRIPTOR_RANGE1 ranges[2];
    // D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE is needed to disable unproper driver optimization.
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, hd->inputs.size(), 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, globalDescHeap.offsetRecord);
    ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, hd->outputs.size(), 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, globalDescHeap.offsetRecord + hd->inputs.size());
    globalDescHeap.offsetRecord += hd->inputs.size() + hd->outputs.size();

    computeRootParameters[0].InitAsDescriptorTable(2, ranges);
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC computeRootSignatureDesc;
    computeRootSignatureDesc.Init_1_1((UINT)computeRootParameters.size(),
        computeRootParameters.data());
#else
    // Prepare Root
    std::vector<CD3DX12_ROOT_PARAMETER1> computeRootParameters(hd->inputs.size() + hd->outputs.size());
    for (int i = 0; i < hd->inputs.size(); ++i)
        computeRootParameters[i].InitAsShaderResourceView(i);
    for (int i = 0; i < hd->outputs.size(); ++i)
        computeRootParameters[hd->inputs.size() + i].InitAsUnorderedAccessView(i);

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC computeRootSignatureDesc;
    computeRootSignatureDesc.Init_1_1(computeRootParameters.size(), computeRootParameters.data());
#endif

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;

    IFE(D3DX12SerializeVersionedRootSignature(&computeRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error));
    IFE(device.pDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_computeRootSignature)));

    computePsoDesc.CS = hd->bytecode;
    computePsoDesc.pRootSignature = m_computeRootSignature.Get();

    IFE(device.pDevice->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&m_computeState)));
    IFE(device.pDevice->CreateCommandAllocator(device.CommandListType, IID_PPV_ARGS(&m_computeCommandAllocator)));
    IFE(device.pDevice->CreateCommandList(0, device.CommandListType, m_computeCommandAllocator.Get(), m_computeState.Get(), IID_PPV_ARGS(&m_computeCommandList)));

    m_computeCommandList->SetComputeRootSignature(m_computeRootSignature.Get());


#ifdef _USE_DESCRIPTOR_HEAP_
    // Prepare Heap Argument Offset
    for (int i = 0; i < hd->inputs.size(); ++i) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
        ZeroMemory(&srvDesc, sizeof(srvDesc));
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = hd->inputs[i].NumElements();
        srvDesc.Buffer.StructureByteStride = hd->inputs[i].TypeSize();
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

        device.pDevice->CreateShaderResourceView(((dx_buffer_t*)buffers[i])->handle.Get(), &srvDesc, globalDescHeap.cpuHandle);
        globalDescHeap.cpuHandle.ptr += globalDescHeap.nDescStep;
    }
    for (int i = 0; i < hd->outputs.size(); ++i) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc;
        ZeroMemory(&uavDesc, sizeof(uavDesc));
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = hd->outputs[i].NumElements();
        uavDesc.Buffer.StructureByteStride = hd->outputs[i].TypeSize();
        device.pDevice->CreateUnorderedAccessView(((dx_buffer_t*)buffers[hd->inputs.size() + i])->handle.Get(), nullptr, &uavDesc, globalDescHeap.cpuHandle);
        globalDescHeap.cpuHandle.ptr += globalDescHeap.nDescStep;
    }

    ID3D12DescriptorHeap* pHeaps[] = { globalDescHeap.heap.Get() };
    m_computeCommandList->SetDescriptorHeaps(1, pHeaps);
    m_computeCommandList->SetComputeRootDescriptorTable(0, globalDescHeap.heap->GetGPUDescriptorHandleForHeapStart());
#else
    for (int i = 0; i < hd->inputs.size(); ++i)
        m_computeCommandList->SetComputeRootShaderResourceView(i, ((dx_buffer_t*)buffers[i])->handle.Get()->GetGPUVirtualAddress());
    for (int i = 0; i < hd->outputs.size(); ++i)
        m_computeCommandList->SetComputeRootUnorderedAccessView(hd->inputs.size() + i, ((dx_buffer_t*)buffers[hd->inputs.size() + i])->handle.Get()->GetGPUVirtualAddress());
#endif


#ifdef _USE_GPU_TIMER_
    int m_nTimerIndex = device.AllocTimerIndex();
    // Set StartTimer here to only consider kernel execution time.
    device.StartTimer(m_computeCommandList.Get(), m_nTimerIndex);
#endif
    m_computeCommandList->Dispatch(hd->block[0], hd->block[1], hd->block[2]);
#ifdef _USE_GPU_TIMER_
    device.StopTimer(m_computeCommandList.Get(), m_nTimerIndex);
#endif

    IFE(m_computeCommandList->Close());
    reservedResources.emplace_back(m_computeCommandList, m_computeCommandAllocator, m_computeRootSignature, m_computeState);
}

__EXPORT__ void dxSynchronize()
{
    std::vector<ID3D12CommandList*> commandList;

    auto sync_list = [&]() -> void {
        if (commandList.empty())
            return;
        device.pCommandQueue->ExecuteCommandLists(commandList.size(), commandList.data());
        device.AwaitExecution();
        commandList.clear();
    };

    for (int i = 0; i < reservedResources.size(); ++i)
    {
        auto &it = reservedResources[i];
        if (it.src != nullptr) // memcpyHtoD
        {
            sync_list();
            device.MapAndCopyToResource(it.deviceCPUSrcX.Get(), it.src, it.bytes);
            commandList.push_back(it.m_computeCommandList.Get());
        }
        else if (it.dst != nullptr) // memcpyDtoH
        {
            commandList.push_back(it.m_computeCommandList.Get());
            sync_list();
            device.MapCopyFromResource(it.deviceCPUSrcX.Get(), it.dst, it.bytes);
        }
        else // launchShader
        {
            commandList.push_back(it.m_computeCommandList.Get());
        }
    }
    sync_list();
    reservedResources.clear();

    device.SyncTimerData();
    auto times = device.GetAllTimes();
    for (auto t : times)
    {
        std::wcout << t << "\n";
    }
    
}

#endif
