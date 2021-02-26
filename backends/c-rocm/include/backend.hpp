// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-rocm): -lamdhip64 -D__HIP_PLATFORM_HCC__ -I/opt/rocm/include -L/opt/rocm/lib
//; eval_flags(c-cuda): -lcuda -lcudart -I/usr/local/cuda/include -L/usr/local/cuda/lib64

#if !defined(__HIP_PLATFORM_HCC__)

#include <cuda.h>
#define BACKEND_TYPE "NVIDIA CUDA"

#else

#include <hip/hip_runtime.h>
#define cuInit hipInit
#define cuMemAlloc hipMalloc
#define cuMemFree hipFree
#define cuModuleLoad hipModuleLoad
#define cuModuleGetFunction hipModuleGetFunction
#define cuLaunchKernel hipModuleLaunchKernel
#define cuMemAllocHost hipHostMalloc
#define cuMemFreeHost hipHostFree
#define cuStreamSynchronize hipStreamSynchronize
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
#define cuDevicePrimaryCtxRetain(x, y) hipSetDevice(y)
#define cuCtxSetCurrent(x) 0
#define BACKEND_TYPE "AMD ROCM"

#endif

namespace ab {

  static int _current_device;
  static std::unordered_map<size_t, std::vector<void*>> _cached_memory;

  void init(int dev) {
    CUcontext ctx;
    // Just one of many methods to set target device id by visiblity
    setenv("CUDA_VISIBLE_DEVICES", std::to_string(dev).c_str(), 1);
    if (0 != cuInit(0) || 0 != cuDevicePrimaryCtxRetain(&ctx, _current_device) || 0 != cuCtxSetCurrent(ctx))
        throw std::runtime_error("GPU device for `" + std::string(BACKEND_TYPE) + "` is not found.\n");
  }

  void* alloc(size_t byteSize, const std::vector<size_t> &shape, const std::string &dtype, const std::string &name) {
    auto &it = _cached_memory[byteSize];
    if (it.size()) {
      auto dptr = it.back();
      it.pop_back();
      return dptr;
    }
    void *dptr = nullptr;
    assert(0 == cuMemAlloc((CUdeviceptr*)&dptr, byteSize));
    return dptr;
  }

  void release(void *dptr, size_t byteSize) {
    auto &it = _cached_memory[byteSize];
    it.push_back(dptr);
  }

  void* moduleLoad(const std::string &source) {
    char temp_name[] = ".antares-module-XXXXXX";
    auto folder = std::string(mkdtemp(temp_name));
    static std::string arch;

#if !defined(__HIP_PLATFORM_HCC__)
    if (!arch.size()) {
      int major, minor;
      assert(0 == cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _current_device));
      assert(0 == cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _current_device));
      arch = std::to_string(major * 10 + minor);
    }
    auto path = folder + "/module.cu";
    auto compile_cmd = "timeout 10s /usr/local/cuda/bin/nvcc " + path + " --fatbin -O2 -gencode arch=compute_" + arch + ",code=sm_" + arch + " -O2 -o " + path + ".out";
#else
    if (!arch.size()) {
      hipDeviceProp_t prop;
      assert(0 == hipGetDeviceProperties(&prop, _current_device));
      arch = std::to_string(prop.gcnArch);
    }
    auto path = folder + "/module.cc";
    auto compile_cmd = "timeout 10s /opt/rocm/bin/hipcc " + path + " --amdgpu-target=gfx" + arch + " --genco -Wno-ignored-attributes -O2 -o " + path + ".out";
#endif
    FILE *fp = fopen(path.c_str(), "w");
    assert(source.size() == fwrite(source.data(), 1, source.size(), fp));
    fclose(fp);
    if (0 != system(compile_cmd.c_str()))
      throw std::runtime_error("Failed to compile module: " + compile_cmd + "\n");

    CUmodule hmod = nullptr;
    assert(0 == cuModuleLoad(&hmod, (path + ".out").c_str()));

    assert(0 == system(("rm -rf " + folder).c_str()));
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
    assert(0 == cuModuleGetFunction(&hfunc, (CUmodule)hModule, fname.c_str()));
    return { hfunc, query("blockIdx.x"), query("blockIdx.y"), query("blockIdx.z"), query("threadIdx.x"), query("threadIdx.y"), query("threadIdx.z") };
  }

  void launchKernel(const std::vector<void*> &hFunc, const std::vector<void*> &krnl_args) {
    std::vector<void* const*> pargs(krnl_args.size());
    for (int i = 0; i < pargs.size(); ++i)
      pargs[i] = &krnl_args[i];
    assert(0 == cuLaunchKernel((CUfunction)hFunc[0], (long)hFunc[1], (long)hFunc[2], (long)hFunc[3], (long)hFunc[4], (long)hFunc[5], (long)hFunc[6], 0, nullptr, (void**)pargs.data(), nullptr));
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize) {
    assert(0 == cuMemcpyHtoDAsync((CUdeviceptr)dptr, hptr, byteSize, nullptr));
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize) {
    assert(0 == cuMemcpyDtoHAsync(hptr, (CUdeviceptr)dptr, byteSize, nullptr));
  }

  void synchronize() {
    assert(0 == cuStreamSynchronize(nullptr));
  }

  void* recordTime() {
    CUevent hEvent;
    assert(0 == cuEventCreate(&hEvent, 0));
    assert(0 == cuEventRecord(hEvent, nullptr));
    return hEvent;
  }

  double convertToElapsedTime(void *hStart, void *hStop) {
    synchronize();

    float ms;
    assert(0 == cuEventElapsedTime(&ms, (CUevent)hStart, (CUevent)hStop));
    assert(0 == cuEventDestroy((CUevent)hStart));
    assert(0 == cuEventDestroy((CUevent)hStop));
    return ms * 1e-3;
  }
}
