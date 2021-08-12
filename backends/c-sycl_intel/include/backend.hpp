// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-sycl_intel): [dpcpp] -ldl -lpthread
//; eval_flags(c-sycl_cuda): [/usr/local/dpcpp-cuda/bin/clang++] -ldl -I/usr/local/dpcpp-cuda/include/sycl -L/usr/local/dpcpp-cuda/lib -lsycl -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda -lpthread -iquote/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -DSYCL_CUDA

#include <CL/sycl.hpp>
#include <dlfcn.h>
#include <pthread.h>
#include <malloc.h>

namespace ab {

  static std::unordered_map<size_t, std::vector<void*>> _cached_memory;
  static sycl::queue _sycl_queue;

  void init(int dev) {
    try {
      if (__BACKEND__ == "c-sycl_intel")
        _sycl_queue = std::move(sycl::queue(sycl::default_selector{}));
      else {
        // for SYCL CUDA, select the i-th GPU device
        int current_dev_id = 0;
        auto platforms = sycl::platform::get_platforms();
        for (auto &p: platforms) {
          auto devices = p.get_devices();
          for (auto &d: devices)
            if (d.is_gpu() && dev == current_dev_id) {
              _sycl_queue = std::move(sycl::queue(d));
              current_dev_id = -1;
              break;
            } else
              current_dev_id++;
          if (current_dev_id < 0)
            break;
        }
      }
    } catch (sycl::exception const &e) {
      std::terminate();
    }
    size_t max_compute_units = _sycl_queue.get_device().get_info<cl::sycl::info::device::max_compute_units>();
    size_t max_work_group_size = _sycl_queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    size_t max_mem_alloc_size = _sycl_queue.get_device().get_info<cl::sycl::info::device::max_mem_alloc_size>();
    size_t local_mem_size = _sycl_queue.get_device().get_info<cl::sycl::info::device::local_mem_size>();
    size_t max_clock_frequency = _sycl_queue.get_device().get_info<cl::sycl::info::device::max_clock_frequency>();
    fprintf(stderr, "    // (SYCL_INFO: SYCL Device Name = %s [%zd, %zd, %zd, %zd, %zd])\n", _sycl_queue.get_device().get_info<sycl::info::device::name>().c_str(),
      max_compute_units, max_work_group_size, max_mem_alloc_size, local_mem_size, max_clock_frequency
    );
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
    if (__BACKEND__ == "c-sycl_intel")
      return memalign(sysconf(_SC_PAGESIZE), byteSize);
    return sycl::malloc_device(byteSize, _sycl_queue);
  }

  void release(void *dptr, size_t byteSize) {
    auto &it = _cached_memory[byteSize];
    it.push_back(dptr);
  }

  void* moduleLoad(const std::string &source) {
    ab_utils::TempFile tempfile("cpp", source);
    auto path = tempfile.get_path();

    if (__BACKEND__ == "c-sycl_intel")
      ab_utils::Process({"dpcpp", path, "-std=c++17", "-lpthread", "-fPIC", "-shared", "-Wno-pass-failed", "-O3", "-ffast-math", "-march=skylake-avx512", "-o", path + ".out"}, 10);
    else {
    std::string gpu_arch = "50"; // Corresponds to the back-end default.
#ifdef SYCL_CUDA
#include "cuda.h"
    int major, minor;
    CHECK_OK(0 == cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0));
    CHECK_OK(0 == cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0));
    gpu_arch = std::to_string(major * 10 + minor);
#endif
      ab_utils::Process({"/usr/local/dpcpp-cuda/bin/clang++", path, "-std=c++17", "-ldl", "-fPIC", "-shared", "-O2", "-I/usr/local/dpcpp-cuda/include/sycl", "-L/usr/local/dpcpp-cuda/lib", "-lsycl", "-fsycl", "-fsycl-targets=nvptx64-nvidia-cuda-sycldevice", "-fsycl-unnamed-lambda", "-Wno-unknown-cuda-version", "-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_" + gpu_arch, "-o", path + ".out"}, 20);  
    }
    path = (path[0] == '/' ? path : "./" + path) + ".out";
    void* hmod = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    CHECK_OK(hmod != nullptr);
    return hmod;
  }

  std::vector<void*> moduleGetFunction(const void *hModule, const std::string &fname, const std::unordered_map<std::string, int> &threads) {
    return { dlsym((void*)hModule, fname.c_str()) };
  }

  void launchKernel(const std::vector<void*> &hFunction, const std::vector<void*> &krnl_args, void *stream) {
    ((void(*)(void*, void* const*))hFunction[0])(&_sycl_queue, krnl_args.data());
    _sycl_queue.wait();
  }

  void synchronize(void *stream) {
    _sycl_queue.wait();
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize, void *stream) {
    ab::synchronize(stream);
    _sycl_queue.memcpy(dptr, hptr, byteSize);
    return;
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize, void *stream) {
    ab::synchronize(stream);
    _sycl_queue.memcpy(hptr, dptr, byteSize);
    return;
    
  }

  void* recordTime(void *stream) {
    ab::synchronize(stream);

    auto pt = new std::chrono::high_resolution_clock::time_point;
    *pt = std::chrono::high_resolution_clock::now();
    return pt;
  }

  double convertToElapsedTime(void *hStart, void *hStop) {
    auto h1 = (std::chrono::high_resolution_clock::time_point*)hStart;
    auto h2 = (std::chrono::high_resolution_clock::time_point*)hStop;

    double et = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(*h2 - *h1).count();
    delete h1, h2;
    return std::max(et, 1e-9);
  }
}
