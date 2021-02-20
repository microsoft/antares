// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-rocm): -lamdhip64 -D__HIP_PLATFORM_HCC__ -I/opt/rocm/include -L/opt/rocm/lib
//; eval_flags(c-cuda): -lcuda -lcudart -I/usr/local/cuda/include -L/usr/local/cuda/lib64

#if !defined(__HIP_PLATFORM_HCC__)
#include <cuda.h>
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
#endif

namespace ab {

  void init() {
    CUcontext ctx;
    const char *dev = getenv("DEV_ID") ? getenv("DEV_ID") : "0";
    setenv("CUDA_VISIBLE_DEVICES", dev, 1);
    if (0 != cuInit(0) || 0 != cuDevicePrimaryCtxRetain(&ctx, 0) || 0 != cuCtxSetCurrent(ctx))
        throw std::runtime_error("GPU device is not found.");
  }

  void* alloc(const tensor_property &tp) {
    static std::unordered_map<std::string, void*> cached_mem;
    void* &dptr = cached_mem[tp.name];
    if (dptr)
      return dptr;
    assert(0 == cuMemAlloc((CUdeviceptr*)&dptr, tp.mem_size()));
    fprintf(stderr, "alloc(%p, `%s`);\n", dptr, tp.name.c_str());
    return dptr;
  }

  void release(void *dptr) {
    fprintf(stderr, "release(%p);\n", dptr);
  }

  void* moduleLoad(const std::string &source) {
    char temp_name[] = ".antares-module-XXXXXX";
    auto folder = std::string(mkdtemp(temp_name));
#if !defined(__HIP_PLATFORM_HCC__)
    auto path = folder + "/module.cu";
    FILE *fp = fopen(path.c_str(), "w");
    assert(source.size() == fwrite(source.data(), 1, source.size(), fp));
    fclose(fp);
    assert(0 == system(("/usr/local/cuda/bin/nvcc " + path + " --fatbin -O2 -gencode arch=compute_70,code=sm_70 -O2 -o " + path + ".out").c_str()));
#else
    auto path = folder + "/module.cc";
    FILE *fp = fopen(path.c_str(), "w");
    assert(source.size() == fwrite(source.data(), 1, source.size(), fp));
    fclose(fp);
    assert(0 == system(("/opt/rocm/bin/hipcc " + path + " --amdgpu-target=gfx906 --genco -Wno-ignored-attributes -O2 -o " + path + ".out").c_str()));
#endif
    CUmodule hmod = nullptr;
    assert(0 == cuModuleLoad(&hmod, (path + ".out").c_str()));

    assert(0 == system(("rm -rf " + folder).c_str()));
    fprintf(stderr, "load(%s)\n", folder.c_str());
    return hmod;
  }

  void* moduleGetFunction(const void *hModule, const std::string &fname) {
    CUfunction hfunc = nullptr;
    assert(0 == cuModuleGetFunction(&hfunc, (CUmodule)hModule, fname.c_str()));
    return hfunc;
  }

  void launchKernel(const void* hFunction, const std::unordered_map<std::string, int> &threads, const std::vector<void*> &krnl_args) {
    auto query = [&](const std::string &axis, int defval = 1) {
      auto it = threads.find(axis);
      if (it == threads.end())
        return defval;
      return it->second;
    };
    std::vector<void* const*> pargs(krnl_args.size());
    for (int i = 0; i < pargs.size(); ++i)
      pargs[i] = &krnl_args[i];
    assert(0 == cuLaunchKernel((CUfunction)hFunction, query("blockIdx.x"), query("blockIdx.y"), query("blockIdx.z"), query("threadIdx.x"), query("threadIdx.y"), query("threadIdx.z"), 0, nullptr, (void**)pargs.data(), nullptr));
    return;

    fprintf(stderr, "launch(");
    for (int i = 0; i < krnl_args.size(); ++i)
      fprintf(stderr, "%p,", krnl_args[i]);
    fprintf(stderr, "\b);\n");
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize) {
    fprintf(stderr, "memcpyHtoD(%zd)\n", byteSize);
    assert(0 == cuMemcpyHtoDAsync((CUdeviceptr)dptr, hptr, byteSize, nullptr));
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize) {
    fprintf(stderr, "memcpyDtoH(%zd)\n", byteSize);
    assert(0 == cuMemcpyDtoHAsync(hptr, (CUdeviceptr)dptr, byteSize, nullptr));
  }

  void synchronize() {
    fprintf(stderr, "synchronize()\n");
    assert(0 == cuStreamSynchronize(nullptr));
  }

  void* recordTime() {
    CUevent hEvent;
    fprintf(stderr, "recordTime()\n");
    assert(0 == cuEventCreate(&hEvent, 0));
    assert(0 == cuEventRecord(hEvent, nullptr));
    return hEvent;
  }

  double convertToElapsedTime(void *hStart, void *hStop) {
    synchronize();
    fprintf(stderr, "convertToElapsedTime()\n");
    float ms;
    assert(0 == cuEventElapsedTime(&ms, (CUevent)hStart, (CUevent)hStop));
    assert(0 == cuEventDestroy((CUevent)hStart));
    assert(0 == cuEventDestroy((CUevent)hStop));
    return ms * 1e-3;
  }
}

