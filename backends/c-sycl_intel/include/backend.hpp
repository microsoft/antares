// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-sycl_intel): -I/opt/intel/oneapi/compiler/latest/linux/include -lsycl -ldl -lpthread -Wno-deprecated-declarations
//; eval_flags(c-sycl_cuda): [/usr/local/dpcpp-cuda/bin/clang++] -ldl -I/usr/local/dpcpp-cuda/include/sycl -L/usr/local/dpcpp-cuda/lib -lsycl -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda -lpthread -iquote/usr/local/cuda/include -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcuda -DSYCL_CUDA

#include <sycl/CL/sycl.hpp>
#include <dlfcn.h>
#include <pthread.h>
#include <malloc.h>
#ifdef SYCL_CUDA
#include "cuda.h"
#endif

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
    // if (__BACKEND__ == "c-sycl_intel")
    //   return memalign(sysconf(_SC_PAGESIZE), byteSize);
    return sycl::malloc_device(byteSize, _sycl_queue);
  }

  void release(void *dptr, size_t byteSize) {
    auto &it = _cached_memory[byteSize];
    it.push_back(dptr);
  }

  std::string moduleCompile(const std::string &source) {
    ab_utils::TempFile tempfile("cpp", source);
    auto path = tempfile.get_path();

    if (__BACKEND__ == "c-sycl_intel")
      ab_utils::Process({"dpcpp", path, "-std=c++17", "-lpthread", "-fPIC", "-shared", "-Wno-pass-failed", "-O3", "-ffast-math", "-march=native", "-o", path + ".out"}, 10);
    else {
      std::string gpu_arch = "50"; // Corresponds to the back-end default.
#ifdef SYCL_CUDA
      int major, minor;
      CHECK_OK(0 == cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0));
      CHECK_OK(0 == cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0));
      gpu_arch = std::to_string(major * 10 + minor);
#endif
      ab_utils::Process({"/usr/local/dpcpp-cuda/bin/clang++", path, "-std=c++17", "-ldl", "-fPIC", "-shared", "-O2", "-I/usr/local/dpcpp-cuda/include/sycl", "-L/usr/local/dpcpp-cuda/lib", "-lsycl", "-fsycl", "-fsycl-targets=nvptx64-nvidia-cuda-sycldevice", "-fsycl-unnamed-lambda", "-Wno-unknown-cuda-version", "-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_" + gpu_arch, "-o", path + ".out"}, 20);
    }

    std::ifstream t(path + ".out", std::ios_base::binary);
    std::string _((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    return _;
  }

  void* moduleLoad(const std::string &binary) {
    ab_utils::TempFile tempfile("out", binary);
    auto path = tempfile.get_path();
    if (path[0] != '/')
      path = "./" + path;
    void* hmod = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    CHECK_OK(hmod != nullptr);
    return hmod;
  }

  std::vector<void*> moduleGetFunction(const void *hModule, const std::string &fname, const std::unordered_map<std::string, int> &threads) {
    auto query = [&](const std::string &axis, long defval = 1) -> void* {
      auto it = threads.find(axis);
      if (it == threads.end())
        return (void*)defval;
      return (void*)(long)it->second;
    };

    std::vector<void*> fdata = { dlsym((void*)hModule, fname.c_str()), query("blockIdx_x") };
    void *item = (void*)query("$", 0);
    if (item) {
      fdata.push_back(item);
      fdata.push_back(query("$$", 1));

      for (int i = 0; ; ++i) {
        void *item = (void*)query("$" + std::to_string(i), 0);
        if (!item)
          break;
        fdata.push_back(item);
      }
    }
    return fdata;
  }

  void launchKernel(const std::vector<void*> &hFunc, const std::vector<void*> &krnl_args, void *stream) {
    long attrs = (long)hFunc[1];
    if (hFunc.size() > 2) {
      attrs = (long)hFunc[3];
      for (int i = 4; i < hFunc.size(); ++i) {
        long val = (long)hFunc[i];
        if (val < 0) continue;

        auto ptr = (size_t)krnl_args[i - 4 + (long)hFunc[2]];
        attrs *= (ptr + val - 1) / val;
      }
      if (!attrs) return;
    }

    ((void(*)(void*, long, void* const*))hFunc[0])(&_sycl_queue, attrs, krnl_args.data());
    if (__BACKEND__ == "c-sycl_intel") // have to sync unlike CUDA
       _sycl_queue.wait();
  }

  void synchronize(void *stream) {
    _sycl_queue.wait();
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize, void *stream) {
    ab::synchronize(stream);
    _sycl_queue.memcpy(dptr, hptr, byteSize);
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize, void *stream) {
    ab::synchronize(stream);
    _sycl_queue.memcpy(hptr, dptr, byteSize);
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
    delete h1; delete h2;
    return std::max(et, 1e-9);
  }
}
