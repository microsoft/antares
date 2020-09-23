// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once
#define _USE_GPU_TIMER_
//#define _USE_DXC_

#include <stdio.h>
#include <stdint.h>
#include <dxgi1_5.h>
#include <d3d12.h>
#include <cassert>
#include <vector>
#include <wrl/client.h>
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <direct.h>
#pragma once

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

#ifdef _USE_DXC_
#include <dxcapi.h>
#pragma comment(lib, "dxcompiler.lib")
#else
#include <d3dcompiler.h>
#pragma comment(lib, "d3dcompiler.lib")
#endif

using namespace std;
using namespace Microsoft::WRL;


#define IFE(x)  ((FAILED(x)) ? (printf("Error-line: (%s) %d\n", __FILE__, __LINE__), _exit(1), 0): 1)

namespace {

    inline const D3D12_COMMAND_QUEUE_DESC D3D12CommandQueueDesc(D3D12_COMMAND_LIST_TYPE type, D3D12_COMMAND_QUEUE_FLAGS flags = D3D12_COMMAND_QUEUE_FLAG_NONE, UINT nodeMask = 0, INT priority = 0)
    {
        D3D12_COMMAND_QUEUE_DESC desc = {
            type,
            priority,
            flags,
            nodeMask
        };
        return desc;
    }

    inline const D3D12_HEAP_PROPERTIES D3D12HeapProperties(
        D3D12_HEAP_TYPE heapType,
        D3D12_CPU_PAGE_PROPERTY pageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL memoryPoolType = D3D12_MEMORY_POOL_UNKNOWN,
        UINT creationNodeMask = 0,
        UINT visibleNodeMask = 0
    )
    {
        D3D12_HEAP_PROPERTIES heapProperties = {
            heapType,
            pageProperty,
            memoryPoolType,
            creationNodeMask,
            visibleNodeMask
        };
        return heapProperties;
    }

    inline const D3D12_RESOURCE_DESC D3D12BufferResourceDesc(
        UINT64 width,
        UINT height = 1,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        UINT64 alignment = 0
    )
    {
        D3D12_RESOURCE_DESC resourceDesc = {
            D3D12_RESOURCE_DIMENSION_BUFFER,
            alignment,
            width,
            height,
            1,
            1,
            DXGI_FORMAT_UNKNOWN,
            {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            flags
        };

        return resourceDesc;
    }

    struct CD3DX12_ROOT_DESCRIPTOR_TABLE1 : public D3D12_ROOT_DESCRIPTOR_TABLE1
    {
        CD3DX12_ROOT_DESCRIPTOR_TABLE1() = default;
        explicit CD3DX12_ROOT_DESCRIPTOR_TABLE1(const D3D12_ROOT_DESCRIPTOR_TABLE1& o) :
            D3D12_ROOT_DESCRIPTOR_TABLE1(o)
        {}
        CD3DX12_ROOT_DESCRIPTOR_TABLE1(
            UINT numDescriptorRanges,
            _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* _pDescriptorRanges)
        {
            Init(numDescriptorRanges, _pDescriptorRanges);
        }

        inline void Init(
            UINT numDescriptorRanges,
            _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* _pDescriptorRanges)
        {
            Init(*this, numDescriptorRanges, _pDescriptorRanges);
        }

        static inline void Init(
            _Out_ D3D12_ROOT_DESCRIPTOR_TABLE1& rootDescriptorTable,
            UINT numDescriptorRanges,
            _In_reads_opt_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* _pDescriptorRanges)
        {
            rootDescriptorTable.NumDescriptorRanges = numDescriptorRanges;
            rootDescriptorTable.pDescriptorRanges = _pDescriptorRanges;
        }
    };

    struct CD3DX12_ROOT_CONSTANTS : public D3D12_ROOT_CONSTANTS
    {
        CD3DX12_ROOT_CONSTANTS() = default;
        explicit CD3DX12_ROOT_CONSTANTS(const D3D12_ROOT_CONSTANTS& o) :
            D3D12_ROOT_CONSTANTS(o)
        {}
        CD3DX12_ROOT_CONSTANTS(
            UINT num32BitValues,
            UINT shaderRegister,
            UINT registerSpace = 0)
        {
            Init(num32BitValues, shaderRegister, registerSpace);
        }

        inline void Init(
            UINT num32BitValues,
            UINT shaderRegister,
            UINT registerSpace = 0)
        {
            Init(*this, num32BitValues, shaderRegister, registerSpace);
        }

        static inline void Init(
            _Out_ D3D12_ROOT_CONSTANTS& rootConstants,
            UINT num32BitValues,
            UINT shaderRegister,
            UINT registerSpace = 0)
        {
            rootConstants.Num32BitValues = num32BitValues;
            rootConstants.ShaderRegister = shaderRegister;
            rootConstants.RegisterSpace = registerSpace;
        }
    };

    struct CD3DX12_ROOT_DESCRIPTOR1 : public D3D12_ROOT_DESCRIPTOR1
    {
        CD3DX12_ROOT_DESCRIPTOR1() = default;
        explicit CD3DX12_ROOT_DESCRIPTOR1(const D3D12_ROOT_DESCRIPTOR1& o) :
            D3D12_ROOT_DESCRIPTOR1(o)
        {}
        CD3DX12_ROOT_DESCRIPTOR1(
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE)
        {
            Init(shaderRegister, registerSpace, flags);
        }

        inline void Init(
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE)
        {
            Init(*this, shaderRegister, registerSpace, flags);
        }

        static inline void Init(
            _Out_ D3D12_ROOT_DESCRIPTOR1& table,
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE)
        {
            table.ShaderRegister = shaderRegister;
            table.RegisterSpace = registerSpace;
            table.Flags = flags;
        }
    };

    struct CD3DX12_ROOT_PARAMETER1 : public D3D12_ROOT_PARAMETER1
    {
        CD3DX12_ROOT_PARAMETER1() = default;
        explicit CD3DX12_ROOT_PARAMETER1(const D3D12_ROOT_PARAMETER1& o) :
            D3D12_ROOT_PARAMETER1(o)
        {}

        static inline void InitAsDescriptorTable(
            _Out_ D3D12_ROOT_PARAMETER1& rootParam,
            UINT numDescriptorRanges,
            _In_reads_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* pDescriptorRanges,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rootParam.ShaderVisibility = visibility;
            CD3DX12_ROOT_DESCRIPTOR_TABLE1::Init(rootParam.DescriptorTable, numDescriptorRanges, pDescriptorRanges);
        }

        static inline void InitAsConstants(
            _Out_ D3D12_ROOT_PARAMETER1& rootParam,
            UINT num32BitValues,
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            rootParam.ShaderVisibility = visibility;
            CD3DX12_ROOT_CONSTANTS::Init(rootParam.Constants, num32BitValues, shaderRegister, registerSpace);
        }

        static inline void InitAsConstantBufferView(
            _Out_ D3D12_ROOT_PARAMETER1& rootParam,
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            rootParam.ShaderVisibility = visibility;
            CD3DX12_ROOT_DESCRIPTOR1::Init(rootParam.Descriptor, shaderRegister, registerSpace, flags);
        }

        static inline void InitAsShaderResourceView(
            _Out_ D3D12_ROOT_PARAMETER1& rootParam,
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            rootParam.ShaderVisibility = visibility;
            CD3DX12_ROOT_DESCRIPTOR1::Init(rootParam.Descriptor, shaderRegister, registerSpace, flags);
        }

        static inline void InitAsUnorderedAccessView(
            _Out_ D3D12_ROOT_PARAMETER1& rootParam,
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParam.ShaderVisibility = visibility;
            CD3DX12_ROOT_DESCRIPTOR1::Init(rootParam.Descriptor, shaderRegister, registerSpace, flags);
        }

        inline void InitAsDescriptorTable(
            UINT numDescriptorRanges,
            _In_reads_(numDescriptorRanges) const D3D12_DESCRIPTOR_RANGE1* pDescriptorRanges,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            InitAsDescriptorTable(*this, numDescriptorRanges, pDescriptorRanges, visibility);
        }

        inline void InitAsConstants(
            UINT num32BitValues,
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            InitAsConstants(*this, num32BitValues, shaderRegister, registerSpace, visibility);
        }

        inline void InitAsConstantBufferView(
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            InitAsConstantBufferView(*this, shaderRegister, registerSpace, flags, visibility);
        }

        inline void InitAsShaderResourceView(
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            InitAsShaderResourceView(*this, shaderRegister, registerSpace, flags, visibility);
        }

        inline void InitAsUnorderedAccessView(
            UINT shaderRegister,
            UINT registerSpace = 0,
            D3D12_ROOT_DESCRIPTOR_FLAGS flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL)
        {
            InitAsUnorderedAccessView(*this, shaderRegister, registerSpace, flags, visibility);
        }
    };

    struct CD3DX12_DESCRIPTOR_RANGE1 : public D3D12_DESCRIPTOR_RANGE1
    {
        CD3DX12_DESCRIPTOR_RANGE1() = default;
        explicit CD3DX12_DESCRIPTOR_RANGE1(const D3D12_DESCRIPTOR_RANGE1& o) :
            D3D12_DESCRIPTOR_RANGE1(o)
        {}
        CD3DX12_DESCRIPTOR_RANGE1(
            D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
            UINT numDescriptors,
            UINT baseShaderRegister,
            UINT registerSpace = 0,
            D3D12_DESCRIPTOR_RANGE_FLAGS flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
            UINT offsetInDescriptorsFromTableStart =
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND)
        {
            Init(rangeType, numDescriptors, baseShaderRegister, registerSpace, flags, offsetInDescriptorsFromTableStart);
        }

        inline void Init(
            D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
            UINT numDescriptors,
            UINT baseShaderRegister,
            UINT registerSpace = 0,
            D3D12_DESCRIPTOR_RANGE_FLAGS flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
            UINT offsetInDescriptorsFromTableStart =
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND)
        {
            Init(*this, rangeType, numDescriptors, baseShaderRegister, registerSpace, flags, offsetInDescriptorsFromTableStart);
        }

        static inline void Init(
            _Out_ D3D12_DESCRIPTOR_RANGE1& range,
            D3D12_DESCRIPTOR_RANGE_TYPE rangeType,
            UINT numDescriptors,
            UINT baseShaderRegister,
            UINT registerSpace = 0,
            D3D12_DESCRIPTOR_RANGE_FLAGS flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
            UINT offsetInDescriptorsFromTableStart =
            D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND)
        {
            range.RangeType = rangeType;
            range.NumDescriptors = numDescriptors;
            range.BaseShaderRegister = baseShaderRegister;
            range.RegisterSpace = registerSpace;
            range.Flags = flags;
            range.OffsetInDescriptorsFromTableStart = offsetInDescriptorsFromTableStart;
        }
    };

    struct CD3DX12_SHADER_BYTECODE : public D3D12_SHADER_BYTECODE
    {
        CD3DX12_SHADER_BYTECODE() = default;
        explicit CD3DX12_SHADER_BYTECODE(const D3D12_SHADER_BYTECODE& o) :
            D3D12_SHADER_BYTECODE(o)
        {}
        CD3DX12_SHADER_BYTECODE(
            _In_ ID3DBlob* pShaderBlob)
        {
            pShaderBytecode = pShaderBlob->GetBufferPointer();
            BytecodeLength = pShaderBlob->GetBufferSize();
        }
        CD3DX12_SHADER_BYTECODE(
            const void* _pShaderBytecode,
            SIZE_T bytecodeLength)
        {
            pShaderBytecode = _pShaderBytecode;
            BytecodeLength = bytecodeLength;
        }
    };

    struct CD3DX12_DEFAULT {};
    extern const DECLSPEC_SELECTANY CD3DX12_DEFAULT D3D12_DEFAULT;

    struct CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC : public D3D12_VERSIONED_ROOT_SIGNATURE_DESC
    {
        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC() = default;
        explicit CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& o) :
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC(o)
        {}
        explicit CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(const D3D12_ROOT_SIGNATURE_DESC& o)
        {
            Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
            Desc_1_0 = o;
        }
        explicit CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(const D3D12_ROOT_SIGNATURE_DESC1& o)
        {
            Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
            Desc_1_1 = o;
        }
        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            Init_1_0(numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
        }
        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER1* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            Init_1_1(numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
        }
        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC(CD3DX12_DEFAULT)
        {
            Init_1_1(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);
        }

        inline void Init_1_0(
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            Init_1_0(*this, numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
        }

        static inline void Init_1_0(
            _Out_ D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc,
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
            desc.Desc_1_0.NumParameters = numParameters;
            desc.Desc_1_0.pParameters = _pParameters;
            desc.Desc_1_0.NumStaticSamplers = numStaticSamplers;
            desc.Desc_1_0.pStaticSamplers = _pStaticSamplers;
            desc.Desc_1_0.Flags = flags;
        }

        inline void Init_1_1(
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER1* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            Init_1_1(*this, numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
        }

        static inline void Init_1_1(
            _Out_ D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc,
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER1* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
            desc.Desc_1_1.NumParameters = numParameters;
            desc.Desc_1_1.pParameters = _pParameters;
            desc.Desc_1_1.NumStaticSamplers = numStaticSamplers;
            desc.Desc_1_1.pStaticSamplers = _pStaticSamplers;
            desc.Desc_1_1.Flags = flags;
        }
    };

    struct CD3DX12_ROOT_SIGNATURE_DESC : public D3D12_ROOT_SIGNATURE_DESC
    {
        CD3DX12_ROOT_SIGNATURE_DESC() = default;
        explicit CD3DX12_ROOT_SIGNATURE_DESC(const D3D12_ROOT_SIGNATURE_DESC& o) :
            D3D12_ROOT_SIGNATURE_DESC(o)
        {}
        CD3DX12_ROOT_SIGNATURE_DESC(
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            Init(numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
        }
        CD3DX12_ROOT_SIGNATURE_DESC(CD3DX12_DEFAULT)
        {
            Init(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);
        }

        inline void Init(
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            Init(*this, numParameters, _pParameters, numStaticSamplers, _pStaticSamplers, flags);
        }

        static inline void Init(
            _Out_ D3D12_ROOT_SIGNATURE_DESC& desc,
            UINT numParameters,
            _In_reads_opt_(numParameters) const D3D12_ROOT_PARAMETER* _pParameters,
            UINT numStaticSamplers = 0,
            _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* _pStaticSamplers = nullptr,
            D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE)
        {
            desc.NumParameters = numParameters;
            desc.pParameters = _pParameters;
            desc.NumStaticSamplers = numStaticSamplers;
            desc.pStaticSamplers = _pStaticSamplers;
            desc.Flags = flags;
        }
    };

    inline HRESULT D3DX12SerializeVersionedRootSignature(
        _In_ const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* pRootSignatureDesc,
        D3D_ROOT_SIGNATURE_VERSION MaxVersion,
        _Outptr_ ID3DBlob** ppBlob,
        _Always_(_Outptr_opt_result_maybenull_) ID3DBlob** ppErrorBlob)
    {
        if (ppErrorBlob != nullptr)
        {
            *ppErrorBlob = nullptr;
        }

        switch (MaxVersion)
        {
        case D3D_ROOT_SIGNATURE_VERSION_1_0:
            switch (pRootSignatureDesc->Version)
            {
            case D3D_ROOT_SIGNATURE_VERSION_1_0:
                return D3D12SerializeRootSignature(&pRootSignatureDesc->Desc_1_0, D3D_ROOT_SIGNATURE_VERSION_1, ppBlob, ppErrorBlob);

            case D3D_ROOT_SIGNATURE_VERSION_1_1:
            {
                HRESULT hr = S_OK;
                const D3D12_ROOT_SIGNATURE_DESC1& desc_1_1 = pRootSignatureDesc->Desc_1_1;

                const SIZE_T ParametersSize = sizeof(D3D12_ROOT_PARAMETER) * desc_1_1.NumParameters;
                void* pParameters = (ParametersSize > 0) ? HeapAlloc(GetProcessHeap(), 0, ParametersSize) : nullptr;
                if (ParametersSize > 0 && pParameters == nullptr)
                {
                    hr = E_OUTOFMEMORY;
                }
                auto pParameters_1_0 = reinterpret_cast<D3D12_ROOT_PARAMETER*>(pParameters);

                if (SUCCEEDED(hr))
                {
                    for (UINT n = 0; n < desc_1_1.NumParameters; n++)
                    {
                        __analysis_assume(ParametersSize == sizeof(D3D12_ROOT_PARAMETER) * desc_1_1.NumParameters);
                        pParameters_1_0[n].ParameterType = desc_1_1.pParameters[n].ParameterType;
                        pParameters_1_0[n].ShaderVisibility = desc_1_1.pParameters[n].ShaderVisibility;

                        switch (desc_1_1.pParameters[n].ParameterType)
                        {
                        case D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS:
                            pParameters_1_0[n].Constants.Num32BitValues = desc_1_1.pParameters[n].Constants.Num32BitValues;
                            pParameters_1_0[n].Constants.RegisterSpace = desc_1_1.pParameters[n].Constants.RegisterSpace;
                            pParameters_1_0[n].Constants.ShaderRegister = desc_1_1.pParameters[n].Constants.ShaderRegister;
                            break;

                        case D3D12_ROOT_PARAMETER_TYPE_CBV:
                        case D3D12_ROOT_PARAMETER_TYPE_SRV:
                        case D3D12_ROOT_PARAMETER_TYPE_UAV:
                            pParameters_1_0[n].Descriptor.RegisterSpace = desc_1_1.pParameters[n].Descriptor.RegisterSpace;
                            pParameters_1_0[n].Descriptor.ShaderRegister = desc_1_1.pParameters[n].Descriptor.ShaderRegister;
                            break;

                        case D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE:
                            const D3D12_ROOT_DESCRIPTOR_TABLE1& table_1_1 = desc_1_1.pParameters[n].DescriptorTable;

                            const SIZE_T DescriptorRangesSize = sizeof(D3D12_DESCRIPTOR_RANGE) * table_1_1.NumDescriptorRanges;
                            void* pDescriptorRanges = (DescriptorRangesSize > 0 && SUCCEEDED(hr)) ? HeapAlloc(GetProcessHeap(), 0, DescriptorRangesSize) : nullptr;
                            if (DescriptorRangesSize > 0 && pDescriptorRanges == nullptr)
                            {
                                hr = E_OUTOFMEMORY;
                            }
                            auto pDescriptorRanges_1_0 = reinterpret_cast<D3D12_DESCRIPTOR_RANGE*>(pDescriptorRanges);

                            if (SUCCEEDED(hr))
                            {
                                for (UINT x = 0; x < table_1_1.NumDescriptorRanges; x++)
                                {
                                    __analysis_assume(DescriptorRangesSize == sizeof(D3D12_DESCRIPTOR_RANGE) * table_1_1.NumDescriptorRanges);
                                    pDescriptorRanges_1_0[x].BaseShaderRegister = table_1_1.pDescriptorRanges[x].BaseShaderRegister;
                                    pDescriptorRanges_1_0[x].NumDescriptors = table_1_1.pDescriptorRanges[x].NumDescriptors;
                                    pDescriptorRanges_1_0[x].OffsetInDescriptorsFromTableStart = table_1_1.pDescriptorRanges[x].OffsetInDescriptorsFromTableStart;
                                    pDescriptorRanges_1_0[x].RangeType = table_1_1.pDescriptorRanges[x].RangeType;
                                    pDescriptorRanges_1_0[x].RegisterSpace = table_1_1.pDescriptorRanges[x].RegisterSpace;
                                }
                            }

                            D3D12_ROOT_DESCRIPTOR_TABLE& table_1_0 = pParameters_1_0[n].DescriptorTable;
                            table_1_0.NumDescriptorRanges = table_1_1.NumDescriptorRanges;
                            table_1_0.pDescriptorRanges = pDescriptorRanges_1_0;
                        }
                    }
                }

                if (SUCCEEDED(hr))
                {
                    CD3DX12_ROOT_SIGNATURE_DESC desc_1_0(desc_1_1.NumParameters, pParameters_1_0, desc_1_1.NumStaticSamplers, desc_1_1.pStaticSamplers, desc_1_1.Flags);
                    hr = D3D12SerializeRootSignature(&desc_1_0, D3D_ROOT_SIGNATURE_VERSION_1, ppBlob, ppErrorBlob);
                }

                if (pParameters)
                {
                    for (UINT n = 0; n < desc_1_1.NumParameters; n++)
                    {
                        if (desc_1_1.pParameters[n].ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
                        {
                            HeapFree(GetProcessHeap(), 0, reinterpret_cast<void*>(const_cast<D3D12_DESCRIPTOR_RANGE*>(pParameters_1_0[n].DescriptorTable.pDescriptorRanges)));
                        }
                    }
                    HeapFree(GetProcessHeap(), 0, pParameters);
                }
                return hr;
            }
            }
            break;

        case D3D_ROOT_SIGNATURE_VERSION_1_1:
            return D3D12SerializeVersionedRootSignature(pRootSignatureDesc, ppBlob, ppErrorBlob);
        }

        return E_INVALIDARG;
    }
}

namespace antares {

    struct D3DDevice
    {
        ComPtr<IDXGIFactory4> pDxgiFactory;
        ComPtr<ID3D12Device1> pDevice;
        ComPtr<ID3D12CommandQueue> pCommandQueue;
        ComPtr<ID3D12CommandAllocator> pCommandAllocator;
        ComPtr<ID3D12Fence> pFence;
        HANDLE event;
        uint64_t fenceValue = 0;
        bool bEnableDebugLayer = false;
        bool bEnableGPUValidation = false;

        // GPU time stamp query doesn't work on some NVIDIA GPUs with specific drivers, so we switch to DIRECT queue.
#ifdef _USE_GPU_TIMER_
        static const D3D12_COMMAND_LIST_TYPE CommandListType = D3D12_COMMAND_LIST_TYPE_DIRECT;
#else
        static const D3D12_COMMAND_LIST_TYPE CommandListType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
#endif

        D3DDevice(bool EnableDebugLayer = false, bool EnableGPUValidation = false)
        {
            bEnableDebugLayer = EnableDebugLayer;
            bEnableGPUValidation = EnableGPUValidation;
        }
        void InitD3DDevice()
        {
            IFE(CreateDXGIFactory1(IID_PPV_ARGS(&pDxgiFactory)));

            if (D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice)))
            {
                ComPtr<IDXGIAdapter3> pAdapter;
                IFE(pDxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&pAdapter)));
                IFE(D3D12CreateDevice(pAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice)));
            }
        }

        void Init()
        {
            // Enable debug layer
            ComPtr<ID3D12Debug> pDebug;
            if (bEnableDebugLayer && SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&pDebug))))
            {
                pDebug->EnableDebugLayer();

                ComPtr<ID3D12Debug1> pDebug1;
                if (bEnableGPUValidation && SUCCEEDED((pDebug->QueryInterface(IID_PPV_ARGS(&pDebug1)))))
                {
                    pDebug1->SetEnableGPUBasedValidation(true);
                }
            }

            InitD3DDevice();

            // Create a command queue
            D3D12_COMMAND_QUEUE_DESC commandQueueDesc = D3D12CommandQueueDesc(CommandListType);
            IFE(pDevice->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(&pCommandQueue)));

            // Create a command allocator
            IFE(pDevice->CreateCommandAllocator(CommandListType, IID_PPV_ARGS(&pCommandAllocator)));

            // Create a CPU-GPU synchronization event
            event = CreateEvent(nullptr, FALSE, FALSE, nullptr);

            // Create a fence to allow GPU to signal upon completion of execution
            IFE(pDevice->CreateFence(fenceValue, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&pFence)));

#ifdef _USE_GPU_TIMER_
            const UINT nMaxTimers = 65536; // Just a big enough number
            // Profiling related resources
            InitProfilingResources(nMaxTimers);
#endif
        }

        void AwaitExecution()
        {
            ++fenceValue;
            IFE(pCommandQueue->Signal(pFence.Get(), fenceValue));

            IFE(pFence->SetEventOnCompletion(fenceValue, event));

            DWORD retVal = WaitForSingleObject(event, INFINITE);
            if (retVal != WAIT_OBJECT_0)
            {
                DebugBreak();
            }
        }

        inline void CreateCommittedResource(
            const D3D12_HEAP_PROPERTIES& heapProperties,
            const D3D12_RESOURCE_DESC& resourceDesc,
            D3D12_RESOURCE_STATES initialState,
            ID3D12Resource** ppResource
        )
        {
            IFE(pDevice->CreateCommittedResource(
                &heapProperties,
                D3D12_HEAP_FLAG_NONE,
                &resourceDesc,
                initialState,
                nullptr,
                IID_PPV_ARGS(ppResource)
            ));
        }
        inline void CreateGPUOnlyResource(UINT64 size, ID3D12Resource** ppResource)
        {
            CreateCommittedResource(
                D3D12HeapProperties(D3D12_HEAP_TYPE_DEFAULT),
                D3D12BufferResourceDesc(size, 1, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_COMMON,
                ppResource
            );
        }
        inline void CreateUploadBuffer(UINT64 size, ID3D12Resource** ppResource)
        {
            CreateCommittedResource(
                D3D12HeapProperties(D3D12_HEAP_TYPE_UPLOAD),
                D3D12BufferResourceDesc(size),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                ppResource
            );
        }

        inline void CreateReadbackBuffer(UINT64 size, ID3D12Resource** ppResource)
        {
            CreateCommittedResource(
                D3D12HeapProperties(D3D12_HEAP_TYPE_READBACK),
                D3D12BufferResourceDesc(size),
                D3D12_RESOURCE_STATE_COPY_DEST,
                ppResource
            );
        }

        inline void CreateDefaultBuffer(UINT64 size, ID3D12Resource** ppResource)
        {
            CreateCommittedResource(
                D3D12HeapProperties(D3D12_HEAP_TYPE_DEFAULT),
                D3D12BufferResourceDesc(size, 1, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_COMMON,
                ppResource
            );
        }

        void MapAndCopyToResource(ID3D12Resource* pResource, const void* pSrc, UINT64 numBytes)
        {
            D3D12_RANGE range = { 0, static_cast<SIZE_T>(numBytes) };
            void* pData;
            IFE(pResource->Map(0, &range, reinterpret_cast<void**>(&pData)));
            memcpy(pData, pSrc, static_cast<SIZE_T>(numBytes));
            pResource->Unmap(0, &range);
        }

        void MapCopyFromResource(ID3D12Resource* pResource, void* pDest, UINT64 numBytes)
        {
            D3D12_RANGE range = { 0, static_cast<SIZE_T>(numBytes) };
            void* pData;
            IFE(pResource->Map(0, &range, reinterpret_cast<void**>(&pData)));
            memcpy(pDest, pData, static_cast<SIZE_T>(numBytes));
            range.End = 0;
            pResource->Unmap(0, &range);
        }
#ifdef _USE_GPU_TIMER_
        // Profiling related resources
    public:
        uint32_t AllocTimerIndex() { return m_nTimers++; }
        void StartTimer(ID3D12GraphicsCommandList* pCmdList, uint32_t nTimerIdx)
        {
            pCmdList->EndQuery(m_pQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, nTimerIdx * 2);
        }
        void StopTimer(ID3D12GraphicsCommandList* pCmdList, uint32_t nTimerIdx)
        {
            pCmdList->EndQuery(m_pQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, nTimerIdx * 2 + 1);
        }
        void SyncTimerData()
        {
            ID3D12CommandList* pCmdLists[] = { m_pResolveCmdList.Get() };
            pCommandQueue->ExecuteCommandLists(1, pCmdLists);
            AwaitExecution();
            uint64_t* pData;
            D3D12_RANGE range = { 0, m_nTimers * 2 * sizeof(uint64_t) };
            IFE(m_pReadBackBuffer->Map(0, &range, reinterpret_cast<void**>(&pData)));
            memcpy_s(m_TimerDataCPU.data(), sizeof(uint64_t) * m_nTimers * 2, pData, sizeof(uint64_t) * m_nTimers * 2);
            m_pReadBackBuffer->Unmap(0, nullptr);
        }

        double GetTime(uint32_t nTimerIdx)
        {
            assert(nTimerIdx < m_nTimers);
            uint64_t TimeStamp1 = m_TimerDataCPU[nTimerIdx * 2];
            uint64_t TimeStamp2 = m_TimerDataCPU[nTimerIdx * 2 + 1];
            return static_cast<double>(TimeStamp2 - TimeStamp1) * m_fGPUTickDelta;
        }

        std::vector<double> GetAllTimes()
        {
            std::vector<double> times;
            times.resize(m_nTimers);
            for (uint32_t i = 0; i < m_nTimers; ++i)
            {
                times[i] = GetTime(i);
            }
            return std::move(times);
        }

        // Lock GPU clock rate for more stable performance measurement.
        // Only works with Win10 developer mode.
        // Note that SetStablePowerState will disable GPU boost and potentially decrease GPU performance.
        // So don't use it in release version application.
        void LockGPUClock()
        {
            auto IsDeveloperModeEnabled = []()
            {
                HKEY hKey;
                if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\AppModelUnlock", 0, KEY_READ, &hKey) != ERROR_SUCCESS)
                {
                    return false;
                }
                DWORD value;
                DWORD nSize = sizeof(DWORD);
                if (RegQueryValueExW(hKey, L"AllowDevelopmentWithoutDevLicense", 0, NULL, reinterpret_cast<LPBYTE>(&value), &nSize) != ERROR_SUCCESS)
                {
                    RegCloseKey(hKey);
                    return false;
                }
                RegCloseKey(hKey);
                return value != 0;
            };
            if (IsDeveloperModeEnabled())
            {
                pDevice->SetStablePowerState(TRUE);
                printf("Win10 developer mode turned on, locked GPU clock.\n");
            }
        }

    private:
        void InitProfilingResources(uint32_t nMaxTimers)
        {
            uint64_t GpuFrequency;
            IFE(pCommandQueue->GetTimestampFrequency(&GpuFrequency));
            m_fGPUTickDelta = 1.0 / static_cast<double>(GpuFrequency);

            D3D12_HEAP_PROPERTIES HeapProps;
            HeapProps.Type = D3D12_HEAP_TYPE_READBACK;
            HeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            HeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            HeapProps.CreationNodeMask = 1;
            HeapProps.VisibleNodeMask = 1;

            D3D12_RESOURCE_DESC BufferDesc;
            BufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            BufferDesc.Alignment = 0;
            BufferDesc.Width = sizeof(uint64_t) * nMaxTimers * 2;
            BufferDesc.Height = 1;
            BufferDesc.DepthOrArraySize = 1;
            BufferDesc.MipLevels = 1;
            BufferDesc.Format = DXGI_FORMAT_UNKNOWN;
            BufferDesc.SampleDesc.Count = 1;
            BufferDesc.SampleDesc.Quality = 0;
            BufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            BufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

            IFE(pDevice->CreateCommittedResource(&HeapProps, D3D12_HEAP_FLAG_NONE, &BufferDesc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_pReadBackBuffer)));
            m_pReadBackBuffer->SetName(L"GpuTimeStamp Buffer");

            D3D12_QUERY_HEAP_DESC QueryHeapDesc;
            QueryHeapDesc.Count = nMaxTimers * 2;
            QueryHeapDesc.NodeMask = 1;
            QueryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
            IFE(pDevice->CreateQueryHeap(&QueryHeapDesc, IID_PPV_ARGS(&m_pQueryHeap)));
            m_pQueryHeap->SetName(L"GpuTimeStamp QueryHeap");

            IFE(pDevice->CreateCommandList(0,
                CommandListType,
                pCommandAllocator.Get(),
                nullptr,
                IID_PPV_ARGS(&m_pResolveCmdList)));
            m_pResolveCmdList->ResolveQueryData(m_pQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, nMaxTimers * 2, m_pReadBackBuffer.Get(), 0);
            m_pResolveCmdList->Close();
            m_nMaxTimers = nMaxTimers;
            m_nTimers = 0;
            m_TimerDataCPU.resize(nMaxTimers * 2);
        }

        Microsoft::WRL::ComPtr<ID3D12QueryHeap> m_pQueryHeap;
        Microsoft::WRL::ComPtr<ID3D12Resource> m_pReadBackBuffer;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_pResolveCmdList;
        double m_fGPUTickDelta = 0.0;
        uint32_t m_nMaxTimers = 0;
        uint32_t m_nTimers = 0;
        std::vector<uint64_t> m_TimerDataCPU;
#endif
    };

#ifdef _USE_DXC_
    class DXCompiler
    {
    public:
        static DXCompiler* Get()
        {
            static DXCompiler sm_compiler;
            return &sm_compiler;

        }

        ComPtr<IDxcBlob> Compile(LPCVOID pText, UINT32 size, LPCWSTR entryName, LPCWSTR profile)
        {
            ComPtr<IDxcBlob> pRet;
            ComPtr<IDxcBlobEncoding> pSrcBlob;
            IFE(m_pLibrary->CreateBlobWithEncodingOnHeapCopy(pText, size, CP_UTF8, &pSrcBlob));
            ComPtr<IDxcOperationResult> pResult;
            // Just set a random name "ShaderFile"
            if (FAILED(m_pCompiler->Compile(pSrcBlob.Get(), L"ShaderFile", entryName, profile, NULL, 0, NULL, 0, NULL, &pResult)))
            {
                if (pResult)
                {
                    ComPtr<IDxcBlobEncoding> pErrorsBlob;
                    if (SUCCEEDED(pResult->GetErrorBuffer(&pErrorsBlob)))
                    {
                        if (pErrorsBlob)
                        {
                            printf("Compilation Error:\n%s\n", (const char*)pErrorsBlob->GetBufferPointer());
                        }
                    }
                }
            }
            else
            {
                pResult->GetResult(&pRet);
            }
            return pRet;
        }
    private:
        DXCompiler()
        {
            IFE(DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&m_pLibrary)));
            IFE(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&m_pCompiler)));
        }
        DXCompiler(const DXCompiler&) = delete;
        DXCompiler& operator=(const DXCompiler&) = delete;

        ComPtr<IDxcLibrary> m_pLibrary;
        ComPtr<IDxcCompiler> m_pCompiler;

    };
#endif

	template<class T>
	std::vector<char> load_data(int rank, size_t num_elements, const T defval = 1) {
		std::vector<char> ret(num_elements * sizeof(T));
		auto hptr = (T*)ret.data();
		for (int i = 0; i < num_elements; ++i)
			hptr[i] = (rank + 1 + i) % 71;
		return std::move(ret);
	}

	class NNfusionTensor {
		ComPtr<ID3D12Resource> deviceGPUSrcX;
		std::vector<size_t> shape;
		size_t type_size;

	public:
		NNfusionTensor(D3DDevice& device, const std::vector<size_t>& shape, size_t type_size): shape(shape), type_size(type_size) {
			device.CreateGPUOnlyResource(type_size * NumElements(), &deviceGPUSrcX);
		}


		size_t NumElements() const {
			return std::accumulate(
				shape.begin(), shape.end(), 1LU, std::multiplies<size_t>());
		}

		size_t TypeSize() const {
			return type_size;
		}

		ComPtr<ID3D12Resource> Data() const {
			return deviceGPUSrcX;
		}

		std::vector<size_t> Shape() const {
			return shape;
		}
	};

	class NNfusionMemcpy {
		ComPtr<ID3D12Resource> deviceGPUSrcX;
		ComPtr<ID3D12Resource> deviceCPUSrcX;
		ComPtr<ID3D12CommandAllocator> pCommandAllocator;
		ComPtr<ID3D12GraphicsCommandList> m_computeCommandList;
		size_t bufferSize;

	public:
		NNfusionMemcpy(D3DDevice& device, std::vector<ID3D12CommandList*>& cmdQueue, NNfusionTensor& dst, const std::vector<char> &src) {
			bufferSize = dst.TypeSize() * dst.NumElements();

			deviceGPUSrcX = dst.Data();
			device.CreateUploadBuffer(bufferSize, &deviceCPUSrcX);
			assert(src.size() <= bufferSize);
			device.MapAndCopyToResource(deviceCPUSrcX.Get(), src.data(), src.size());

			IFE(device.pDevice->CreateCommandAllocator(device.CommandListType, IID_PPV_ARGS(&pCommandAllocator)));
			IFE(device.pDevice->CreateCommandList(0, device.CommandListType, pCommandAllocator.Get(), nullptr, IID_PPV_ARGS(&m_computeCommandList)));
			m_computeCommandList->CopyResource(deviceGPUSrcX.Get(), deviceCPUSrcX.Get());
			m_computeCommandList->Close();

			cmdQueue.push_back(Launch());
		}

		NNfusionMemcpy(D3DDevice& device, std::vector<ID3D12CommandList*>& cmdQueue, void *dst, NNfusionTensor& src) {
			bufferSize = src.TypeSize() * src.NumElements();

			deviceGPUSrcX = src.Data();
			device.CreateReadbackBuffer(bufferSize, &deviceCPUSrcX);

			IFE(device.pDevice->CreateCommandAllocator(device.CommandListType, IID_PPV_ARGS(&pCommandAllocator)));
			IFE(device.pDevice->CreateCommandList(0, device.CommandListType, pCommandAllocator.Get(), nullptr, IID_PPV_ARGS(&m_computeCommandList)));
			m_computeCommandList->CopyResource(deviceCPUSrcX.Get(), deviceGPUSrcX.Get());
			m_computeCommandList->Close();

			cmdQueue.push_back(Launch());
		}

		ID3D12GraphicsCommandList* Launch() {
			return m_computeCommandList.Get();
		}

		template <class T>
		double PrintStageBuffer(D3DDevice& device)
		{
			assert(bufferSize % sizeof(T) == 0);
			std::vector<T> dst(bufferSize / sizeof(T));
			device.MapCopyFromResource(deviceCPUSrcX.Get(), dst.data(), bufferSize);
			double ans = 0;
			for (int i = 0, ceof = 1; i < dst.size(); ++i, ceof = (ceof + 1) % 83) {
				ans += double(dst[i]) * ceof;
			}
			return ans;
		}
	};

	class NNfusionOperator {
		ComPtr<ID3D12GraphicsCommandList> m_computeCommandList;

		ComPtr<ID3D12RootSignature> m_computeRootSignature;
#ifdef _USE_DXC_
        ComPtr<IDxcBlob> computeShader;
#else
		ComPtr<ID3DBlob> computeShader;
#endif
		ComPtr<ID3D12PipelineState> m_computeState;
		ComPtr<ID3D12CommandAllocator> computeCommandAllocator;
		D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc;

		LPCSTR hlsl_source;
#ifdef _USE_GPU_TIMER_
		int m_nTimerIndex = -1;
#endif

	public:
		NNfusionOperator(D3DDevice& device, std::vector<ID3D12CommandList*>& cmdQueue, const std::vector<NNfusionTensor>& inputs, const std::vector<NNfusionTensor>& outputs, const std::vector<size_t> &threads, LPCSTR hlsl_source)
				: hlsl_source(hlsl_source) {

#define _USE_DECRIPTOR_HEAP_

#ifdef _USE_DECRIPTOR_HEAP_

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

			std::vector<UINT> argsOffset;
			// Prepare Heap Argument Offset
			for (int i = 0; i < inputs.size(); ++i) {
				D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
				ZeroMemory(&srvDesc, sizeof(srvDesc));
				srvDesc.Format = DXGI_FORMAT_UNKNOWN;
				srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
				srvDesc.Buffer.FirstElement = 0;
				srvDesc.Buffer.NumElements = inputs[i].NumElements();
				srvDesc.Buffer.StructureByteStride = inputs[i].TypeSize();
				srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

				device.pDevice->CreateShaderResourceView(inputs[i].Data().Get(), &srvDesc, globalDescHeap.cpuHandle);
				globalDescHeap.cpuHandle.ptr += globalDescHeap.nDescStep;
				argsOffset.push_back(globalDescHeap.offsetRecord++);
				assert(globalDescHeap.offsetRecord <= MAX_HEAP_SIZE);
			}
			for (int i = 0; i < outputs.size(); ++i) {
				D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc;
				ZeroMemory(&uavDesc, sizeof(uavDesc));
				uavDesc.Format = DXGI_FORMAT_UNKNOWN;
				uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
				uavDesc.Buffer.FirstElement = 0;
				uavDesc.Buffer.NumElements = outputs[i].NumElements();
				uavDesc.Buffer.StructureByteStride = outputs[i].TypeSize();
				device.pDevice->CreateUnorderedAccessView(outputs[i].Data().Get(), nullptr, &uavDesc, globalDescHeap.cpuHandle);
				globalDescHeap.cpuHandle.ptr += globalDescHeap.nDescStep;
				argsOffset.push_back(globalDescHeap.offsetRecord++);
				assert(globalDescHeap.offsetRecord <= MAX_HEAP_SIZE);
			}

			// Prepare Root
			std::vector<CD3DX12_ROOT_PARAMETER1> computeRootParameters(1);
			CD3DX12_DESCRIPTOR_RANGE1 ranges[2];
			// D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE is needed to disable unproper driver optimization.
			ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, inputs.size(), 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, argsOffset[0]);
			ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, outputs.size(), 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, argsOffset[inputs.size()]);

			computeRootParameters[0].InitAsDescriptorTable(2, ranges);
			CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC computeRootSignatureDesc;
			computeRootSignatureDesc.Init_1_1((UINT)computeRootParameters.size(),
				computeRootParameters.data());
#else
			// Prepare Root
			std::vector<CD3DX12_ROOT_PARAMETER1> computeRootParameters(inputs.size() + outputs.size());
			for (int i = 0; i < inputs.size(); ++i)
				computeRootParameters[i].InitAsShaderResourceView(i);
			for (int i = 0; i < outputs.size(); ++i)
				computeRootParameters[inputs.size() + i].InitAsUnorderedAccessView(i);

			CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC computeRootSignatureDesc;
			computeRootSignatureDesc.Init_1_1(computeRootParameters.size(), computeRootParameters.data());
#endif

			ComPtr<ID3DBlob> signature;
			ComPtr<ID3DBlob> error;

			IFE(D3DX12SerializeVersionedRootSignature(&computeRootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error));
			IFE(device.pDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_computeRootSignature)));

#ifdef _USE_DXC_
            // Use cs_6_0 since dxc only supports cs_6_0 or higher shader models.
            computeShader = DXCompiler::Get()->Compile(hlsl_source, strlen(hlsl_source), L"CSMain", L"cs_6_0");
            computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader->GetBufferPointer(), computeShader->GetBufferSize());
#else
            IFE(D3DCompile(hlsl_source, strlen(hlsl_source), NULL, NULL, NULL, "CSMain", "cs_5_1", 0, 0, &computeShader, NULL));
            computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.Get());
#endif
            computePsoDesc.pRootSignature = m_computeRootSignature.Get();

			IFE(device.pDevice->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&m_computeState)));
			IFE(device.pDevice->CreateCommandAllocator(device.CommandListType, IID_PPV_ARGS(&computeCommandAllocator)));
			IFE(device.pDevice->CreateCommandList(0, device.CommandListType, computeCommandAllocator.Get(), m_computeState.Get(), IID_PPV_ARGS(&m_computeCommandList)));


			m_computeCommandList->SetComputeRootSignature(m_computeRootSignature.Get());


#ifdef _USE_DECRIPTOR_HEAP_
			ID3D12DescriptorHeap* pHeaps[] = { globalDescHeap.heap.Get() };
			m_computeCommandList->SetDescriptorHeaps(1, pHeaps);
			m_computeCommandList->SetComputeRootDescriptorTable(0, globalDescHeap.heap->GetGPUDescriptorHandleForHeapStart());
#else
			for (int i = 0; i < inputs.size(); ++i)
				m_computeCommandList->SetComputeRootShaderResourceView(i, inputs[i].Data()->GetGPUVirtualAddress());
			for (int i = 0; i < outputs.size(); ++i)
				m_computeCommandList->SetComputeRootUnorderedAccessView(inputs.size() + i, outputs[i].Data()->GetGPUVirtualAddress());
#endif

#ifdef _USE_GPU_TIMER_
			m_nTimerIndex = device.AllocTimerIndex();
			// Set StartTimer here to only consider kernel execution time.
			device.StartTimer(m_computeCommandList.Get(), m_nTimerIndex);
#endif
			m_computeCommandList->Dispatch(threads[0], threads[1], threads[2]);
#ifdef _USE_GPU_TIMER_
			device.StopTimer(m_computeCommandList.Get(), m_nTimerIndex);
#endif

			IFE(m_computeCommandList->Close());

			cmdQueue.push_back(Launch());
		}

		ID3D12GraphicsCommandList* Launch() {
			return m_computeCommandList.Get();
		}

#ifdef _USE_GPU_TIMER_
		int TimerIndex() { return m_nTimerIndex; }
#endif
	};
}
