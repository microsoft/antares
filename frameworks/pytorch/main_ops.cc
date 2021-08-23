// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <torch/extension.h>
#include <vector>
#include <string>

#include "execute_module.hpp"

static std::vector<std::unique_ptr<ExecutionModule>> module_manager;

void custom_op_forward(std::vector<torch::Tensor> tensors, int custom_key, const std::string& source)
{
  if (module_manager.size() <= custom_key)
    module_manager.resize(custom_key + 1);

  if (module_manager[custom_key] == nullptr)
  {
    int ord = 0;
#if defined(ANTARES_CUDA)
    cuCtxGetDevice(&ord);
#elif defined(ANTARES_ROCM)
    hipGetDevice(&ord);
#endif
    ab::init(ord);
    module_manager[custom_key] = std::make_unique<ExecutionModule>(source);
    return;
  }

  auto &module = module_manager[custom_key];
  const auto input_size = module->global_inputs.size();
  const auto output_size = module->global_outputs.size();
  CHECK_EQ(input_size + output_size, tensors.size());

  std::vector<void*> args(tensors.size());
  int arg_size = args.size();
  for (int i = 0; i < arg_size; ++i)
    args[i] = (void*)tensors[i].data_ptr();

  module->compute(args.data());

#if !defined(ANTARES_CUDA) && !defined(ANTARES_ROCM)
  ab::synchronize(0);
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &custom_op_forward, "custom forward (GPU)");
}
