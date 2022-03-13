// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <torch/extension.h>
#include <vector>
#include <string>
#include <unordered_map>

#include "execute_module.hpp"

static std::vector<std::string> local_sources;
static std::unordered_map<int, std::unique_ptr<ExecutionModule>> module_manager;

static int custom_op_inject(const std::string& source)
{
  int custom_key = local_sources.size();
  local_sources.push_back(source);
  return custom_key;
}

static void custom_op_forward(int custom_key, std::vector<torch::Tensor> tensors)
{
  int dev = tensors[0].device().index(), comp_key = custom_key * 32 + dev;
  auto it = module_manager.find(comp_key);
  if (it == module_manager.end()) {
    CHECK_EQ(true, custom_key < local_sources.size());
    module_manager[comp_key] = std::make_unique<ExecutionModule>(local_sources[custom_key]);
    it = module_manager.find(comp_key);
  }

  auto &module = module_manager[comp_key];
  const auto input_size = module->global_inputs.size();
  const auto output_size = module->global_outputs.size();
  CHECK_EQ(input_size + output_size, tensors.size());

  std::vector<void*> args(tensors.size());
  int arg_size = args.size();
  for (int i = 0; i < arg_size; ++i)
    args[i] = (void*)tensors[i].data_ptr();

  module->compute(args.data());

#if !defined(HIP_VERSION) && !defined(CUDA_VERSION)
  ab::synchronize(0);
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inject", &custom_op_inject, "custom inject");
  m.def("forward", &custom_op_forward, "custom forward");
}
