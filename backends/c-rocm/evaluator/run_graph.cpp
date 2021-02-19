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
        if (next > it)
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

    size_t mem_size() const {
        return element_size() * type_size();
    }
};

struct kernel_property {
  std::vector<std::string> args;
  std::unordered_map<std::string, int> threads;
  void* hFunction;
};

std::vector<tensor_property> parse_properties(const std::string &encoded_inputs) {
    if (encoded_inputs.size() == 0)
      return {};
    std::vector<tensor_property> ret;
    for (auto it: ssplit(encoded_inputs, "], ")) {
      auto props = ssplit(it, ":");
      tensor_property tp;
      tp.name = props[0];
      props = ssplit(props[1], "[");
      tp.dtype = props[0];

      for (auto dim: ssplit(props[1], ", "))
        tp.shape.push_back(std::atol(dim.c_str()));
      ret.push_back(std::move(tp));
    }
    return std::move(ret);
}

#include "backend.hpp"

struct ExecutionModule {
  std::vector<tensor_property> global_inputs, global_outputs;
  std::unordered_map<std::string, tensor_property> local_tensors;
  std::map<std::string, kernel_property> local_kernels;

  std::string backend;

  void *hModule;

  ExecutionModule(const std::string &source) {
    auto encoded_params = get_between(source, "// GLOBALS: ", "\n");
    auto params = ssplit(encoded_params, " -> ");
    global_inputs = parse_properties(params[0]), global_outputs = parse_properties(params[1]);

    encoded_params = get_between(source, "// TENSORS: ", "\n");
    for (auto &tensor: parse_properties(encoded_params)) {
      local_tensors[tensor.name] = tensor;
    }

    backend = get_between(source, "// BACKEND: ", " (");
    // fprintf(stderr, "%s\n", backend.c_str());

    hModule = ab::moduleLoad(source);

    auto kernel_slices = ssplit(source, "-------\n");
    for (int i = 1; i < kernel_slices.size(); ++i) {
      auto name = get_between(kernel_slices[i], "// LOCAL: ", " - ");
      auto &kp = local_kernels[name];
      for (auto arg: ssplit(get_between(kernel_slices[i], " - ", "\n"), ", "))
        kp.args.push_back(arg.substr(0, arg.find('(')));
      // for (auto &it: kp.args)
      //   fprintf(stderr, "  arg(%s) = %s\n", name.c_str(), it.c_str());
      int idx = 0, next;
      while (next = kernel_slices[i].find("// [thread_extent] ", idx), next >= 0) {
        auto thread_key = get_between(kernel_slices[i], "] ", " = ", next);
        auto thread_val = std::atoi(get_between(kernel_slices[i], " = ", "\n", next).c_str());
        auto &val = kp.threads[thread_key];
        if (val > 0 && val != thread_val)
          throw std::runtime_error(("Multiple `" + thread_key + "` extents conflict in values: " + std::to_string(val) + " v.s. " + std::to_string(thread_val) + ";\n").c_str());
        val = thread_val;
        idx = next + 1;
      }

      kp.hFunction = ab::moduleGetFunction(hModule, name);
    }
  }

  size_t compute(void **args) {
    std::unordered_map<std::string, int> tensor_used;
    for (int i = 0; i < global_inputs.size(); ++i)
      ++tensor_used[global_inputs[i].name];
    int nodeCnt = local_kernels.size();
    for (auto it = --local_kernels.end(); nodeCnt > 0; --it, --nodeCnt) {
      if (nodeCnt == local_kernels.size()) {
        for (int i = 0; i < it->second.args.size() - global_outputs.size(); ++i)
          ++tensor_used[it->second.args[i]];
      } else {
        for (int i = 0; i < it->second.args.size() - 1; ++i)
          ++tensor_used[it->second.args[i]];
      }
    }
    std::unordered_map<std::string, void*> tensor_memory;
    for (int i = 0; i < global_inputs.size(); ++i)
      tensor_memory[global_inputs[i].name] = args[i];
    for (int i = 0; i < global_outputs.size(); ++i)
      tensor_memory[global_outputs[i].name] = args[i + global_inputs.size()];

    for (auto it = local_kernels.begin(); ++nodeCnt <= local_kernels.size(); ++it) {
      const std::string &name = it->first;
      if (nodeCnt != local_kernels.size()) {
        auto &arg_name = it->second.args.back();
        auto &memptr = tensor_memory[arg_name];
        assert(memptr == nullptr);
        memptr = ab::alloc(local_tensors[arg_name]);
      }
      std::vector<void*> krnl_args;
      for (auto &arg: it->second.args)
        krnl_args.push_back(tensor_memory[arg]);

      ab::launchKernel(it->second.hFunction, it->second.threads, krnl_args);
      // get_funciton(); launch_kernel(krnl_args.data());

      int num_inputs = it->second.args.size() - (nodeCnt != local_kernels.size() ? 1 : global_outputs.size());
      for (int i = 0; i < num_inputs; ++i)
        if (--tensor_used[it->second.args[i]] == 0) {
          ab::release(tensor_memory[it->second.args[i]]);
        }
    }
    return 0;
  }
};

static void *timeout_monitor(void *arg) {
    sleep(30);
    fprintf(stderr, "[FATAL] Time limit exceeded for this evaluation.\n");
    exit(1);
}

int main(int argc, char** argv)
{
    pthread_t p_timeout_monitor;
    pthread_create(&p_timeout_monitor, NULL, timeout_monitor, NULL);
    pthread_detach(p_timeout_monitor);

    ab::init();

    std::ifstream t("my_kernel.cc");
    std::string source((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    t.close();

#if 1
    ExecutionModule gm(source);
    std::vector<void*> global_args;
    for (int i = 0; i < gm.global_inputs.size(); ++i) {
      auto &it = gm.global_inputs[i];
      void *dptr = ab::alloc(it);
      global_args.push_back(dptr);

      std::vector<char> hptr(it.mem_size());
      size_t size = it.element_size();
      if (it.dtype == "int32") {
        for (size_t x = 0; x < size; ++x)
          ((int*)hptr.data())[x] = (x + i + 1) % 71;
      } else if (it.dtype == "float32") {
        for (size_t x = 0; x < size; ++x)
          ((float*)hptr.data())[x] = (x + i + 1) % 71;
      } else {
        size_t byte_size = size * it.type_size();
        for (size_t x = 0; x < byte_size / sizeof(int); ++x)
          ((int*)hptr.data())[x] = (x + i + 1) % 71;
        for (size_t x = byte_size - byte_size % sizeof(int); x < byte_size; x++)
          ((char*)hptr.data())[x] = 1;
      }
      ab::memcpyHtoD(dptr, hptr.data(), hptr.size());
      ab::synchronize();
    }
    for (auto &it: gm.global_outputs) {
      void *dptr = ab::alloc(it);
      global_args.push_back(dptr);
    }

    gm.compute(global_args.data());

    for (int i = 0; i < gm.global_outputs.size(); ++i) {
      auto &it = gm.global_outputs[i];
      void *dptr = global_args[gm.global_inputs.size() + i];

      std::vector<char> hptr(it.mem_size());
      ab::memcpyDtoH(hptr.data(), dptr, hptr.size());
      ab::synchronize();

      size_t byte_size = it.mem_size();
      double digest = 0.0;
      if (it.dtype == "int32") {
        for (size_t x = 0; x < byte_size / sizeof(int); ++x)
          digest += (x + 1) % 83 * ((int*)hptr.data())[x];
      } else {
        for (size_t x = 0; x < byte_size / sizeof(float); ++x)
          digest += (x + 1) % 83 * ((float*)hptr.data())[x];
        for (size_t x = byte_size - byte_size % sizeof(int); x < byte_size; x++)
          digest += ((char*)hptr.data())[x];
      }
      printf("\n- K/%d: %.10e\n", i, digest);
    }

    {
      auto x = ab::recordTime();
      gm.compute(global_args.data());
      auto y = ab::recordTime();
      ab::synchronize();

      double tpr = ab::convertToElapsedTime(x, y);
      const char *expected_timeout = getenv("EXPECTED_TIMEOUT");
      if (expected_timeout && *expected_timeout && tpr > std::atof(expected_timeout))
        throw std::runtime_error(("Time limit exceeded: " + std::to_string(tpr) + " v.s. (expected) " + expected_timeout).c_str());

      int num_runs = (int)std::max(1LU, std::min(10000LU, (unsigned long)(1.0 / tpr)));
      tpr = 0.0f;
      x = ab::recordTime();
      for (int i = 0; i < num_runs; ++i)
        gm.compute(global_args.data());
      y = ab::recordTime();
      tpr = ab::convertToElapsedTime(x, y) / num_runs;
      printf("\n- TPR: %g\n", tpr);
    }
    return 0;
#else
    struct kernel_prop {
      std::vector<void*> args;
      std::vector<void**> pargs;
      std::vector<int> kthreads;
      CUfunction hfunc;
    };

    std::unordered_map<std::string, void*> mediate_ptrs;
    std::map<std::string, kernel_prop> kernels;

    auto mediate_params = get_between(source, "// TENSORS: ", "\n");
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

    auto encoded_params = get_between(source, "// GLOBALS: ", "\n");
    auto params = ssplit(encoded_params, " -> ");
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
#endif
    return 0;
}
