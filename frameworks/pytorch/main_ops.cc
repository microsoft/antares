// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <torch/extension.h>
#include <vector>
#include <string>

#include "execute_module.hpp"

struct ModuleConfig {
	std::vector<std::vector<int64_t>> output_shapes;
    std::vector<std::decay_t<decltype(torch::kFloat32)>> output_dtypes;
    std::unique_ptr<ExecutionModule> module;
};

static std::vector<ModuleConfig> module_manager;

std::vector<torch::Tensor> custom_op_forward(std::vector<torch::Tensor> inputs, int custom_key, const std::string& source)
{
  if (module_manager.size() <= custom_key)
    module_manager.resize(custom_key + 1);

  if (module_manager[custom_key].module == nullptr)
  {
    int ord = 0;
#if defined(ANTARES_CUDA)
    cuCtxGetDevice(&ord);
#elif defined(ANTARES_ROCM)
    hipGetDevice(&ord);
#endif
    ab::init(ord);
    module_manager[custom_key] = {{}, {}, std::make_unique<ExecutionModule>(source)};

    for (auto &output: module_manager[custom_key].module->global_outputs) {
      std::vector<int64_t> shape;
      for (auto &it: output.shape)
        shape.push_back(it);
      std::decay_t<decltype(torch::kFloat32)> dtype = torch::kInt32;
      if (output.dtype == "float32")
        dtype = torch::kFloat32;
      else if (output.dtype == "float16")
        dtype = torch::kFloat16;
      else if (output.dtype == "float64")
        dtype = torch::kFloat64;
      else if (output.dtype == "int64")
        dtype = torch::kInt64;
      else if (output.dtype == "int16")
        dtype = torch::kInt16;
      else if (output.dtype == "int8")
        dtype = torch::kInt8;
      else
        CHECK_EQ(dtype, torch::kInt32);
      module_manager[custom_key].output_shapes.push_back(std::move(shape));
      module_manager[custom_key].output_dtypes.push_back(dtype);
    }
    return {};
  }

  auto &gm = module_manager[custom_key];
  const auto input_size = gm.module->global_inputs.size();
  const auto output_size = gm.module->global_outputs.size();
  CHECK_EQ(input_size, inputs.size());

  std::vector<void*> args(input_size + output_size);
  for (int i = 0; i < input_size; ++i)
    args[i] = (void*)inputs[i].data_ptr();

  std::vector<torch::Tensor> outputs;
  for (int i = 0; i < output_size; ++i) {
    outputs.push_back(torch::empty(gm.output_shapes[i],
      torch::TensorOptions()
        .dtype(gm.output_dtypes[i])
        .device(inputs[0].device().type(), inputs[0].device().index())
        .layout(torch::kStrided)
        .requires_grad(false)
    ));
    args[input_size + i] = (void*)outputs[i].data_ptr();
  }

  gm.module->compute(args.data());
#if !defined(ANTARES_CUDA) && !defined(ANTARES_ROCM)
  ab::synchronize(0);
#endif
  return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &custom_op_forward, "custom forward (GPU)");
}
