// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-rocm): -lamdhip64 -D__HIP_PLATFORM_HCC__ -I/opt/rocm/include -L/opt/rocm/lib
//; eval_flags(c-cuda): -lcuda -lcudart -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs

#if !defined(__HIP_PLATFORM_HCC__)
#include <cuda.h>
#else
#include <hip/hip_runtime.h>
#define cuInit hipInit
#define cuMemAlloc hipMalloc
#define cuMemFree hipFree
#define cuModuleLoad hipModuleLoad
#define cuModuleLoadData hipModuleLoadData
#define cuModuleUnload hipModuleUnload
#define cuModuleGetFunction hipModuleGetFunction
#define cuLaunchKernel hipModuleLaunchKernel
#define cuMemAllocHost hipHostMalloc
#define cuMemFreeHost hipHostFree
#define cuStreamSynchronize hipStreamSynchronize
#define cuCtxSynchronize hipDeviceSynchronize
#define cuMemcpyHtoDAsync hipMemcpyHtoDAsync
#define cuMemcpyDtoHAsync hipMemcpyDtoHAsync
#define CUdeviceptr hipDeviceptr_t
#define CUmodule hipModule_t
#define CUfunction hipFunction_t
#define CUevent hipEvent_t
#define cuEventElapsedTime hipEventElapsedTime
#define cuEventCreate hipEventCreateWithFlags
#define cuEventDestroy hipEventDestroy
#define cuEventRecord hipEventRecord
#define CUcontext long
#define cuDevicePrimaryCtxRetain(x, y) (*(x) = (CUcontext)((long)(y)), 0)
#define cuCtxSetCurrent(x) hipSetDevice((long)(x))
#define CUstream hipStream_t
#endif


namespace ab {

  static int _current_device;
  static std::unordered_map<size_t, std::vector<void*>> _cached_memory;

  void init(int dev) {
    CUcontext ctx;
    // Just one of many methods to set target device id by visiblity
    setenv("CUDA_VISIBLE_DEVICES", std::to_string(dev).c_str(), 1);
    if (0 != cuInit(0) || 0 != cuDevicePrimaryCtxRetain(&ctx, _current_device) || 0 != cuCtxSetCurrent(ctx))
        throw std::runtime_error("GPU device is not found.\n");
  }

  void finalize() {
  }

  void* alloc(size_t byteSize, const std::vector<size_t> &shape, const std::string &dtype, const std::string &name) {
    auto &it = _cached_memory[byteSize];
    if (it.size()) {
      auto dptr = it.back();
      it.pop_back();
      return dptr;
    }
    void *dptr = nullptr;
    if (byteSize)
      CHECK_OK(0 == cuMemAlloc((CUdeviceptr*)&dptr, byteSize));
    else
      dptr = (void*)1LU;
    return dptr;
  }

  void release(void *dptr, size_t byteSize) {
    auto &it = _cached_memory[byteSize];
    it.push_back(dptr);
  }

  std::string moduleCompile(const std::string &source) {
    ab_utils::TempFile tempfile("cu", source);
    auto &path = tempfile.get_path();

#if !defined(__HIP_PLATFORM_HCC__)
    static std::string _gpu_arch;
    if (!_gpu_arch.size()) {
      int major, minor;
      CHECK_OK(0 == cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _current_device));
      CHECK_OK(0 == cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _current_device));
      _gpu_arch = std::to_string(major * 10 + minor);
    }
    std::vector<std::string> compile_args = {"/usr/local/cuda/bin/nvcc", path, "--fatbin", "-w", "-O2", "-o", (path + ".out")};
    static std::vector<std::string> compat = {"52", "53", "60", "61", "62", "70", "72", "75", "80"};
    if (getenv("CUDA_ALL_ARCH") == nullptr)
      compat = { _gpu_arch };
    for (int i = 0; i < compat.size(); ++i) {
      compile_args.push_back("-gencode");
      compile_args.push_back("arch=compute_" + compat[i] + ",code=sm_" + compat[i]);
    }
#else
    static std::string _gpu_arch;
    if (!_gpu_arch.size()) {
      hipDeviceProp_t prop;
      CHECK_OK(0 == hipGetDeviceProperties(&prop, _current_device));
      _gpu_arch = "gfx" + std::to_string(prop.gcnArch);
    }

    std::vector<std::string> codes = { get_between(source, "\n#define __AMDGFX__ ", "\n") };
    if (codes[0].size() == 0)
      codes[0] = _gpu_arch;

    std::vector<std::string> compile_args = {"/opt/rocm/bin/hipcc", path, "--genco", "-O2", "-Wno-ignored-attributes", "-o", (path + ".out")};
    for (auto &code: codes)
      compile_args.push_back("--amdgpu-target=" + code);
#endif

    ab_utils::Process(compile_args, 30);
    return file_read((path + ".out").c_str());
  }

  void* moduleLoad(const std::string &binary) {
    CUmodule hmod = nullptr;
    CHECK_OK(0 == cuModuleLoadData(&hmod, binary.data()));
    return hmod;
  }

  std::vector<void*> moduleGetFunction(const void *hModule, const std::string &fname, const std::unordered_map<std::string, int> &threads) {
    auto query = [&](const std::string &axis, long defval = 1) -> void* {
      auto it = threads.find(axis);
      if (it == threads.end())
        return (void*)defval;
      return (void*)(long)it->second;
    };

    CUfunction hfunc = nullptr;
    CHECK_OK(0 == cuModuleGetFunction(&hfunc, (CUmodule)hModule, fname.c_str()));
    std::vector<void*> fdata = { hfunc, query("blockIdx.x"), query("blockIdx.y"), query("blockIdx.z"), query("threadIdx.x"), query("threadIdx.y"), query("threadIdx.z") };

    void *item = query("$", 0);
    if (item) {
      fdata.push_back(item);
      fdata.push_back(query("$$", 1));

      for (int i = 0; ; ++i) {
        void *item = query("$" + std::to_string(i), 0);
        if (!item)
          break;
        fdata.push_back(item);
      }
    }
    return fdata;
  }

  void launchKernel(std::vector<void*> &hFunc, const std::vector<void*> &krnl_args, void *stream) {
    std::vector<void*> pargs(krnl_args.size());
    for (int i = 0; i < krnl_args.size(); ++i)
      pargs[i] = (void*)&krnl_args[i];

    if (hFunc.size() > 7) {
      long attrs = (long)hFunc[8];
      for (int i = 9; i < hFunc.size(); ++i) {
        long val = (long)hFunc[i];
        if (val < 0) continue;

        auto ptr = (int*)pargs[i - 9 + (long)hFunc[7]];
        attrs *= (*ptr + val - 1) / val;
      }
      hFunc[1] = (void*)attrs;
      if (!hFunc[1]) return;
    }

    CHECK_OK(0 == cuLaunchKernel((CUfunction)hFunc[0], (long)hFunc[1], (long)hFunc[2], (long)hFunc[3], (long)hFunc[4], (long)hFunc[5], (long)hFunc[6],
      0, (CUstream)stream, (void**)pargs.data(), nullptr));
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize, void *stream) {
    CHECK_OK(0 == cuMemcpyHtoDAsync((CUdeviceptr)dptr, hptr, byteSize, (CUstream)stream));
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize, void *stream) {
    CHECK_OK(0 == cuMemcpyDtoHAsync(hptr, (CUdeviceptr)dptr, byteSize, (CUstream)stream));
  }

  void synchronize(void *stream) {
    CHECK_OK(0 == cuStreamSynchronize((CUstream)stream));
  }

  void* recordTime(void *stream) {
    CUevent hEvent;
    CHECK_OK(0 == cuEventCreate(&hEvent, 0));
    CHECK_OK(0 == cuEventRecord(hEvent, (CUstream)stream));
    return hEvent;
  }

  double convertToElapsedTime(void *hStart, void *hStop) {
    CHECK_OK(0 == cuCtxSynchronize());

    float ms;
    CHECK_OK(0 == cuEventElapsedTime(&ms, (CUevent)hStart, (CUevent)hStop));
    CHECK_OK(0 == cuEventDestroy((CUevent)hStart));
    CHECK_OK(0 == cuEventDestroy((CUevent)hStop));
    return ms * 1e-3;
  }
}
