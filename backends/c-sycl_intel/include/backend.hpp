// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-sycl_intel): [dpcpp] -ldl -lpthread
//; eval_flags(c-sycl_cuda): [/usr/local/dpcpp-cuda/bin/clang++] -ldl -I/usr/local/dpcpp-cuda/include/sycl -L/usr/local/dpcpp-cuda/lib -lsycl -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda -lpthread

#include <CL/sycl.hpp>
#include <dlfcn.h>
#include <pthread.h>
#include <malloc.h>

namespace ab {

  static std::unordered_map<size_t, std::vector<void*>> _cached_memory;
  static sycl::queue _sycl_queue;

  void init(int dev) {
    try {
       _sycl_queue = std::move(sycl::queue(sycl::default_selector{}));
    } catch (sycl::exception const &e) {
      std::terminate();
    }
    size_t max_compute_units = _sycl_queue.get_device().get_info<cl::sycl::info::device::max_compute_units>();
    size_t max_work_group_size = _sycl_queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    size_t max_mem_alloc_size = _sycl_queue.get_device().get_info<cl::sycl::info::device::max_mem_alloc_size>();
    size_t local_mem_size = _sycl_queue.get_device().get_info<cl::sycl::info::device::local_mem_size>();
    size_t max_clock_frequency = _sycl_queue.get_device().get_info<cl::sycl::info::device::max_clock_frequency>();
    fprintf(stderr, "    (SYCL_INFO: SYCL Device Name = %s [%zd, %zd, %zd, %zd, %zd])\n", _sycl_queue.get_device().get_info<sycl::info::device::name>().c_str(),
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

    // FIXME: Only handle dtype = float32 in this version (SYCL buffer is bind to datatype at compile time?)
    CHECK_OK(dtype == "float32" && byteSize % sizeof(float) == 0);
    return new cl::sycl::buffer<float>((float*)malloc(byteSize), cl::sycl::range<1>(byteSize / sizeof(float)), cl::sycl::property::buffer::context_bound(_sycl_queue.get_context()));
  }

  void release(void *dptr, size_t byteSize) {
    auto &it = _cached_memory[byteSize];
    it.push_back(dptr);
  }

  void* moduleLoad(const std::string &source) {
    ab_utils::TempFile tempfile("cpp", source);
    auto path = tempfile.get_path();

    if (__BACKEND__ == "c-sycl_intel")
      ab_utils::Process({"dpcpp", path, "-std=c++17", "-lpthread", "-fPIC", "-shared", "-Wno-pass-failed", "-O2", "-o", path + ".out"}, 10);
    else
      ab_utils::Process({"/usr/local/dpcpp-cuda/bin/clang++", path, "-std=c++17", "-ldl", "-fPIC", "-shared", "-O2", "-I/usr/local/dpcpp-cuda/include/sycl", "-L/usr/local/dpcpp-cuda/lib", "-lsycl", "-fsycl", "-fsycl-targets=nvptx64-nvidia-cuda-sycldevice", "-fsycl-unnamed-lambda", "-Wno-unknown-cuda-version", "-o", path + ".out"}, 20);

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
    if (__BACKEND__ == "c-sycl_intel")
      _sycl_queue.wait();
  }

  void synchronize(void *stream) {
    _sycl_queue.wait();
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize, void *stream) {
    if (__BACKEND__ == "c-sycl_intel") {
      ab::synchronize(stream);
      memcpy(dptr, hptr, byteSize);
      return;
    }

    auto &buff = *((cl::sycl::buffer<float>*)dptr);
    _sycl_queue.submit([&](cl::sycl::handler& cgh) {
      auto d_data = buff.get_access<cl::sycl::access::mode::discard_write>(cgh);
      cgh.copy(hptr, d_data);
    });
    ab::synchronize(stream);
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize, void *stream) {
    if (__BACKEND__ == "c-sycl_intel") {
      ab::synchronize(stream);
      memcpy(hptr, dptr, byteSize);
      return;
    }

    auto &buff = *((cl::sycl::buffer<float>*)dptr);
    _sycl_queue.submit([&](cl::sycl::handler& cgh) {
      auto d_data = buff.get_access<cl::sycl::access::mode::read>(cgh);
      cgh.copy(d_data, hptr);
    });
    ab::synchronize(stream);
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
