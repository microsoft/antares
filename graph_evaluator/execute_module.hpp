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
#include <cstring>
#include <functional>
#include <numeric>

#if !defined(_WIN64) || defined(__MINGW64__)
#include <pthread.h>
#include <unistd.h>
#endif

#define CHECK_OK(x)  ((x) ? 1 : (fprintf(stderr, "[CheckFail] %s:%d\n", __FILE__, __LINE__), exit(1), 0))

namespace ab_utils {

  class TempFile {
    std::string file_path;

  public:
    TempFile(const std::string &extension_name, const std::string &file_content) {
      // FIXME: Be careful it's not thread-safe in shared context. Fortunately, no shared context used in current version.
      this->file_path = ".antares-module-tempfile." + extension_name;

      FILE *fp = fopen(this->file_path.c_str(), "w");
      CHECK_OK(fp != nullptr);
      CHECK_OK(file_content.size() == fwrite(file_content.data(), 1, file_content.size(), fp));
      fclose(fp);
    }

    ~TempFile() {
      remove(this->file_path.c_str());
    }

    const std::string& get_path() {
      return this->file_path;
    }
  };

  class Process {

  public:
    Process(const std::vector<std::string> &cmd_args, int timeout_sec = -1) {
      std::string system_cmd;
      for (int i = 0; i < cmd_args.size(); ++i) {
        if (i)
          system_cmd += ' ';
        for (int j = 0; j < cmd_args[i].size(); ++j)
#if defined(__linux__)
          if (cmd_args[i][j] == '\'')
            system_cmd += "'\"'\"'";
          else
#endif
            system_cmd += cmd_args[i][j];
      }

#if defined(__linux__)
      std::string warp_cmd = "timeout " + std::to_string(timeout_sec) + " sh -c '" + system_cmd + "'";
#else
      std::string warp_cmd = system_cmd;
#endif
      if (0 != system(warp_cmd.c_str()))
        throw std::runtime_error("Failed to execute command: sh -c '" + system_cmd + "'\n");
    }
  };
}

#include "backend.hpp"

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

std::vector<std::string> ssplit(const std::string &str, const std::string &sub, bool allow_empty = false) {
    std::vector<std::string> ret;
    int it = 0, next;
    while (next = str.find(sub, it), next >= 0) {
        if (next > it || allow_empty)
            ret.push_back(str.substr(it, next - it));
        it = next + sub.size();
    }
    if (it < str.size() || allow_empty)
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
                CHECK_OK((bits & 7) == 0);
                return bits >> 3;
            }
        }
        throw std::runtime_error(("Unrecognized type size for `" + dtype + "`").c_str());
    }

    size_t mem_size() {
        if (!_mem_size)
            _mem_size = element_size() * type_size();
        return _mem_size;
    }
private:
    int _mem_size = 0;
};

struct kernel_property {
  std::string fname;
  std::vector<std::string> in_args, out_args;
  std::unordered_map<std::string, int> threads;
  std::vector<void*> hFunction;
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

void *allocate_tensor(tensor_property &tp) {
  size_t align_size = tp.mem_size();
  return ab::alloc(align_size, tp.shape, tp.dtype, tp.name);
}

struct ExecutionModule {
  std::vector<tensor_property> global_inputs, global_outputs;
  std::unordered_map<std::string, tensor_property> local_tensors;
  std::vector<kernel_property> local_kernels;

  std::string backend;

  void *hModule;
  bool debug_output;

  ExecutionModule(std::string source) {
    debug_output = getenv("AB_DEBUG") && *getenv("AB_DEBUG") ? atoi(getenv("AB_DEBUG")) : 0;

    static const char file_proto[] = "file://";

    if (0 == strncmp(source.c_str(), file_proto, sizeof(file_proto) - 1)) {
      FILE *fp = fopen(source.c_str() + sizeof(file_proto) - 1, "rb");
      fseek(fp, 0, SEEK_END);
      size_t filesize = ftell(fp);
      fseek(fp, 0, SEEK_SET);
      source.resize(filesize);
      CHECK_OK(filesize = fread((void*)source.data(), 1, filesize, fp));
      fclose(fp);
    }

    auto encoded_params = get_between(source, "// GLOBALS: ", "\n");
    auto params = ssplit(encoded_params, " -> ", true);
    global_inputs = parse_properties(params[0]), global_outputs = parse_properties(params[1]);

    backend = get_between(source, "// BACKEND: ", " (");
    // fprintf(stderr, "%s\n", backend.c_str());

    hModule = ab::moduleLoad(source);

    auto kernel_slices = ssplit(source, "-------\n");
    for (int i = 1; i < kernel_slices.size(); ++i) {
      auto name = get_between(kernel_slices[i], "// LOCAL: ", " -- ");
      local_kernels.push_back(kernel_property{});
      auto &kp = local_kernels[local_kernels.size() - 1];
      kp.fname = name;
      auto inputs_outputs = ssplit(get_between(kernel_slices[i], " -- ", "\n"), " -> ", true);
      auto local_inputs = parse_properties(inputs_outputs[0]);
      auto local_outputs = parse_properties(inputs_outputs[1]);

      for (auto &arg: local_inputs) {
        kp.in_args.push_back(arg.name);
        local_tensors[arg.name] = arg;
      }
      for (auto &arg: local_outputs) {
        kp.out_args.push_back(arg.name);
        local_tensors[arg.name] = arg;
      }
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

      kp.hFunction = ab::moduleGetFunction(hModule, kp.fname, kp.threads);
    }
  }

  size_t compute(void **args) {
    std::unordered_map<std::string, int> tensor_used;
    for (int i = 0; i < global_inputs.size(); ++i)
      ++tensor_used[global_inputs[i].name];
    for (auto it = local_kernels.begin(); it != local_kernels.end(); ++it) {
      for (int i = 0; i < it->in_args.size(); ++i)
          ++tensor_used[it->in_args[i]];
    }
    std::unordered_map<std::string, void*> tensor_memory;
    for (int i = 0; i < global_inputs.size(); ++i)
      tensor_memory[global_inputs[i].name] = args[i];
    for (int i = 0; i < global_outputs.size(); ++i)
      tensor_memory[global_outputs[i].name] = args[i + global_inputs.size()];

    int nodeCnt = 0;
    for (auto it = local_kernels.begin(); ++nodeCnt <= local_kernels.size(); ++it) {
      const std::string &name = it->fname;
      if (nodeCnt != local_kernels.size()) {
        CHECK_OK(it->out_args.size() == 1);
        auto &arg_name = it->out_args[0];
        auto &memptr = tensor_memory[arg_name];
        CHECK_OK(memptr == nullptr);
        memptr = allocate_tensor(local_tensors.find(arg_name)->second);
      }
      std::vector<void*> krnl_args;
      for (auto &arg: it->in_args)
        krnl_args.push_back(tensor_memory[arg]);
      for (auto &arg: it->out_args)
        krnl_args.push_back(tensor_memory[arg]);

      ab::launchKernel(it->hFunction, krnl_args);

      int num_inputs = it->in_args.size();
      for (int i = 0; i < num_inputs; ++i)
        if (--tensor_used[it->in_args[i]] == 0) {
          ab::release(tensor_memory[it->in_args[i]], local_tensors[it->in_args[i]].mem_size());
        }

      if (debug_output) {
        for (auto &arg: it->out_args) {
          char d[32];
          ab::memcpyDtoH(d, tensor_memory[arg], sizeof(d));
          ab::synchronize();
          if (local_tensors[arg].dtype == "float32")
            fprintf(stderr, "[DEBUG] %s(%s) = %g, %g, %g, %g ..\n", arg.c_str(), local_tensors[arg].dtype.c_str(), ((float*)d)[0], ((float*)d)[1], ((float*)d)[2], ((float*)d)[3]);
          else if (local_tensors[arg].dtype == "float64")
            fprintf(stderr, "[DEBUG] %s(%s) = %g, %g, %g, %g ..\n", arg.c_str(), local_tensors[arg].dtype.c_str(), ((double*)d)[0], ((double*)d)[1], ((double*)d)[2], ((double*)d)[3]);
          else if (local_tensors[arg].dtype == "int32")
            fprintf(stderr, "[DEBUG] %s(%s) = %d, %d, %d, %d ..\n", arg.c_str(), local_tensors[arg].dtype.c_str(), ((int*)d)[0], ((int*)d)[1], ((int*)d)[2], ((int*)d)[3]);
          else
            fprintf(stderr, "[DEBUG] %s(%s) = %016x, %016x, %016x, %016x ..\n", arg.c_str(), local_tensors[arg].dtype.c_str(), ((int*)d)[0], ((int*)d)[1], ((int*)d)[2], ((int*)d)[3]);
        }
      }
    }
    if (debug_output)
      fprintf(stderr, "[DEBUG] =======================\n");

    return 0;
  }
};
