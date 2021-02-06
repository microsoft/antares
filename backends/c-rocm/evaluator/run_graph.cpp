// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <vector>
#include <cassert>
#include <cstring>
#include <functional>
#include <numeric>
#include <pthread.h>
#include <unistd.h>

#if !defined(__HIPCC__)
#include <cuda.h>
#else
#include <hip/hip_runtime.h>
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
#define cuEventRecord hipEventRecord
#endif

std::string get_between(const std::string &str, const std::string &begin, const std::string &end, int start_idx = 0, const std::string &def_ret = "") {
    if (start_idx < 0)
        return def_ret;
    int at = str.find(begin, start_idx);
    if (at < 0)
        return def_ret;
    at += begin.size();
    int next = str.find(end, at);
    if (next < at)
        return def_ret;
    return str.substr(at, next - at);
}

std::vector<std::string> ssplit(const std::string &str, const std::string &sub) {
    std::vector<std::string> ret;
    int it = 0, next;
    while (next = str.find(sub, it), next >= 0) {
        ret.push_back(str.substr(it, next - it));
        it = next + sub.size();
    }
    ret.push_back(str.substr(it));
    return std::move(ret);
}

struct tensor_property {
    std::string name, dtype;
    std::vector<size_t> shape;

    size_t element_size() const {
        return std::accumulate(shape.begin(), shape.end(), (size_t)1L, std::multiplies<size_t>());
    }

    int type_size() const {
        for (int i = dtype.size() - 1; i >= 0; --i) {
            if (!isdigit(dtype[i])) {
                int bits = std::atoi(dtype.substr(i + 1).c_str());
                assert((bits & 7) == 0);
                return bits >> 3;
            }
        }
        throw std::runtime_error(("Unrecognized type size for `" + dtype + "`").c_str());
    }

    size_t mem_size() {
        return element_size() * type_size();
    }
};

std::vector<tensor_property> parse_properties(const std::string &encoded_inputs) {
    if (encoded_inputs.size() == 0)
      return {};
    std::vector<tensor_property> ret;
    for (auto it: ssplit(encoded_inputs, ",")) {
      auto props = ssplit(it, "/");
      tensor_property tp;
      auto sshape = ssplit(props[0], "-");
      for (auto d: sshape)
        tp.shape.push_back(std::atol(d.c_str()));
      tp.dtype = props[1];
      tp.name = props[2];
      ret.push_back(tp);
    }
    return std::move(ret);
}

void *timeout_monitor(void *arg) {
    sleep(30);
    fprintf(stderr, "[FATAL] Time limit exceeded for this evaluation.\n");
    exit(1);
}

int main(int argc, char** argv)
{
    pthread_t p_timeout_monitor;
    pthread_create(&p_timeout_monitor, NULL, timeout_monitor, NULL);
    pthread_detach(p_timeout_monitor);

#if !defined(__HIPCC__)
    CUcontext ctx;
    if (0 != cuInit(0) || 0 != cuDevicePrimaryCtxRetain(&ctx, 0) || 0 != cuCtxSetCurrent(ctx))
        throw std::runtime_error("GPU device for CUDA is not found.");
#else
    if (0 != hipSetDevice(0))
        throw std::runtime_error("GPU device for ROCM is not found.");
#endif

    std::ifstream t("my_kernel.cc");
    std::string source((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    t.close();

    struct kernel_prop {
      std::vector<void*> args;
      std::vector<void**> pargs;
      std::vector<int> kthreads;
      CUfunction hfunc;
    };

    std::unordered_map<std::string, void*> mediate_ptrs;
    std::map<std::string, kernel_prop> kernels;

    auto mediate_params = get_between(source, "// -- ", "\n");
    for (auto mediate: ssplit(mediate_params, "], ")) {
      auto parts = ssplit(mediate, ":");
      auto name = parts[0];
      parts = ssplit(parts[1], "[");
      tensor_property tp = {name, parts[0]};
      for (auto dim: ssplit(parts[1], ", ")) {
        tp.shape.push_back(std::atoi(dim.c_str()));
      }
      assert(0 == cuMemAlloc((CUdeviceptr*)&mediate_ptrs[name], tp.mem_size()));
      // fprintf(stderr, ">> (%s) -> (%p)\n", name.c_str(), mediate_ptrs[name]);
    }
    auto get_extent = [](const std::string &thread_name, const std::string &source, int start_idx = 0) -> int {
      return std::atoi(get_between(source, "// [thread_extent] " + thread_name + " =", "\n", start_idx, "1").c_str());
    };
    int idx = 0, next, tail;
    while (next = source.find(" void ", idx), next >= 0) {
      tail = source.find("\n}", next);
      assert(tail >= 0);
      auto kernel_func = source.substr(next, tail + 2 - next);
      auto func_name = get_between(kernel_func, " void ", "(");
      auto args_str = get_between(kernel_func, "(", ")");
      auto &prop = kernels[func_name];
      for (auto item: ssplit(args_str, ", ")) {
        auto arg_name = ssplit(item, " ").back();
        if (arg_name.size() > 2 && arg_name[0] == '_' && arg_name[1] == '_')
          arg_name = arg_name.substr(2);
        auto ptr = mediate_ptrs[arg_name];
        assert(ptr != nullptr);
        prop.args.push_back(ptr);
      }
      prop.pargs.resize(prop.args.size());
      for (int i = 0; i < prop.args.size(); ++i)
        prop.pargs[i] = &prop.args[i];

      auto &thv = prop.kthreads;
      thv.push_back(get_extent("blockIdx.x", kernel_func));
      thv.push_back(get_extent("blockIdx.y", kernel_func));
      thv.push_back(get_extent("blockIdx.z", kernel_func));
      thv.push_back(get_extent("threadIdx.x", kernel_func));
      thv.push_back(get_extent("threadIdx.y", kernel_func));
      thv.push_back(get_extent("threadIdx.z", kernel_func));
      // fprintf(stderr, "------- %s %s (%d %d %d, %d %d %d)\n", func_name.c_str(), args_str.c_str(), thv[0], thv[1], thv[2], thv[3], thv[4], thv[5]);

      idx = next + 1;
    }

    auto encoded_params = get_between(source, "///", "\n");
    auto params = ssplit(encoded_params, ":");
    auto inputs = parse_properties(params[0]), outputs = parse_properties(params[1]);

    std::vector<void*> h_args, d_args;
    for (int i = 0; i < inputs.size(); ++i) {
      auto &it = inputs[i];
      std::pair<void*, void*> ptrs = {nullptr, mediate_ptrs[it.name]};
      assert(0 == cuMemAllocHost(&ptrs.first, it.mem_size()));
      h_args.push_back(ptrs.first);
      d_args.push_back(ptrs.second);

      size_t size = it.element_size();
      if (it.dtype == "int32") {
        for (size_t x = 0; x < size; ++x)
          ((int*)(ptrs.first))[x] = (x + i + 1) % 71;
      } else if (it.dtype == "float32") {
        for (size_t x = 0; x < size; ++x)
          ((float*)(ptrs.first))[x] = (x + i + 1) % 71;
      } else {
        size_t byte_size = size * it.type_size();
        for (size_t x = 0; x < byte_size / sizeof(int); ++x)
          ((int*)(ptrs.first))[x] = (x + i + 1) % 71;
        for (size_t x = byte_size - byte_size % sizeof(int); x < byte_size; x++)
          ((char*)(ptrs.first))[x] = 1;
      }
      if (ptrs.first != ptrs.second)
        assert(0 == cuMemcpyHtoDAsync((CUdeviceptr)ptrs.second, ptrs.first, size * it.type_size(), nullptr));
    }
    for (auto it: outputs) {
      std::pair<void*, void*> ptrs = {nullptr, mediate_ptrs[it.name]};
      assert(0 == cuMemAllocHost(&ptrs.first, it.mem_size()));
      h_args.push_back(ptrs.first);
      d_args.push_back(ptrs.second);

      memset(ptrs.first, 0, it.element_size() * it.type_size());
      if (ptrs.first != ptrs.second)
        assert(0 == cuMemcpyHtoDAsync((CUdeviceptr)ptrs.second, ptrs.first, it.element_size() * it.type_size(), nullptr));
    }

    CUmodule hmod;
    CUfunction hfunc;
    assert(0 == cuModuleLoad(&hmod, "my_kernel.out"));
    for (auto &it: kernels)
      assert(0 == cuModuleGetFunction(&it.second.hfunc, hmod, it.first.c_str()));

    auto launch_kernel = [&]() -> void {
      for (auto &it: kernels) {
        auto &p = it.second;
        assert(0 == cuLaunchKernel(p.hfunc, p.kthreads[0], p.kthreads[1], p.kthreads[2], p.kthreads[3], p.kthreads[4], p.kthreads[5], 0, nullptr, (void**)p.pargs.data(), nullptr));
      }
    };

    launch_kernel();

    for (int c = 0; c < outputs.size(); ++c) {
      size_t byte_size = outputs[c].mem_size();
      if (h_args[inputs.size() + c] != d_args[inputs.size() + c])
        assert(0 == cuMemcpyDtoHAsync(h_args[inputs.size() + c], (CUdeviceptr)d_args[inputs.size() + c], byte_size, nullptr));
    }
    assert(0 == cuStreamSynchronize(nullptr));

    for (int c = 0; c < outputs.size(); ++c) {
      size_t byte_size = outputs[c].mem_size();
      double digest = 0.0;
      if (outputs[c].dtype == "int32") {
        for (size_t i = 0; i < byte_size / sizeof(int); ++i)
          digest += (i + 1) % 83 * ((int*)h_args[inputs.size() + c])[i];
      } else {
        for (size_t i = 0; i < byte_size / sizeof(float); ++i)
          digest += (i + 1) % 83 * ((float*)h_args[inputs.size() + c])[i];
        for (size_t i = byte_size - byte_size % sizeof(int); i < byte_size; i++)
          digest += ((char*)h_args[inputs.size() + c])[i];
      }
      printf("- K/%d: %.10e\n", c, digest);
    }

    CUevent hStart, hStop;
    float ms;
    assert(0 == cuEventCreate(&hStart, 0));
    assert(0 == cuEventCreate(&hStop, 0));

    assert(0 == cuEventRecord(hStart, nullptr));
    launch_kernel();
    assert(0 == cuEventRecord(hStop, nullptr));
    assert(0 == cuStreamSynchronize(nullptr));
    assert(0 == cuEventElapsedTime(&ms, hStart, hStop));
    float tpr = ms * 1e-3;

    const char *expected_timeout = getenv("EXPECTED_TIMEOUT");
    if (expected_timeout && *expected_timeout && tpr > std::atof(expected_timeout)) {
        throw std::runtime_error(("Time limit exceeded: " + std::to_string(tpr) + " v.s. (expected) " + expected_timeout).c_str());
    }

    int num_runs = std::max(1, std::min(10000, int(1.0 / tpr)));
    bool flush_global_memory = (getenv("FLUSH_MEM") != nullptr);

    tpr = 0.0f;
    if (flush_global_memory) {
      num_runs = 10;
      for (int i = 0; i < num_runs; ++i) {
        for (int j = 0; j < inputs.size(); ++j)
           assert(0 == cuMemcpyHtoDAsync((CUdeviceptr)d_args[j], h_args[j], inputs[j].mem_size(), nullptr));
        assert(0 == cuEventRecord(hStart, nullptr));
        launch_kernel();
        assert(0 == cuEventRecord(hStop, nullptr));
        assert(0 == cuStreamSynchronize(nullptr));
        assert(0 == cuEventElapsedTime(&ms, hStart, hStop));
        tpr += ms * 1e-3;
      }
      tpr /= num_runs;
    } else {
      assert(0 == cuEventRecord(hStart, nullptr));
      for (int i = 0; i < num_runs; ++i)
        launch_kernel();
      assert(0 == cuEventRecord(hStop, nullptr));
      assert(0 == cuStreamSynchronize(nullptr));
      assert(0 == cuEventElapsedTime(&ms, hStart, hStop));
      tpr = ms * 1e-3 / num_runs;
    }
    printf("- TPR: %g\n", tpr);

    for (auto &it: h_args)
      assert(0 == cuMemFreeHost(it));
    for (auto &it: d_args)
      assert(0 == cuMemFree((CUdeviceptr)it));
    return 0;
}
