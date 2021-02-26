// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <torch/extension.h>
#include <vector>
#include <string>
#include <map>

#include "execute_module.hpp"

static std::map<std::string, std::shared_ptr<ExecutionModule>> module_manager;

std::vector<torch::Tensor> custom_op_forward(std::vector<torch::Tensor> inputs,
                                             const std::string& source,
                                             const std::string& source_path,
                                             const std::string& hash,
                                             const std::vector<std::string>& meta_inputs,
                                             const std::vector<std::string>& meta_outputs)
{
  auto it = module_manager.find(hash);
  if (it == module_manager.end())
  {
    module_manager[hash] = std::make_shared<ExecutionModule>(source);
    it = module_manager.find(hash);
  }
  auto gm = it->second;

  std::vector<void*> args;
  args.resize(meta_inputs.size() + meta_outputs.size());

  std::vector<std::vector<int64_t>> output_shapes;
  for (auto &meta_output: meta_outputs) {
    int idx = meta_output.find('[');
    CHECK_EQ(true, idx > 0);
    std::vector<int64_t> shape_builder;
    for (int i = idx + 1, j = i + 1; j <= meta_output.size(); ++j) {
      if (j == meta_output.size() || meta_output[j] == ',')
        shape_builder.push_back(std::atoi(meta_output.c_str() + i)), i = j + 2, ++j;
    }
    output_shapes.push_back(std::move(shape_builder));
  }

  std::vector<torch::Tensor> outputs;
  outputs.resize(meta_outputs.size());
  auto options =
    torch::TensorOptions()
      .dtype(inputs[0].dtype())
      .device(inputs[0].device().type(), inputs[0].device().index())
      .layout(torch::kStrided)
      .requires_grad(true);

  for (int i = 0; i < outputs.size(); ++i)
    outputs[i] = torch::zeros(output_shapes[i], options);

  for (int i = 0; i < inputs.size(); ++i)
    args[i] = (void*)inputs[i].data_ptr();

  for (int i = 0; i < meta_outputs.size(); ++i)
    args[meta_inputs.size() + i] = (void*)outputs[i].data_ptr();

  gm->compute(args.data());
  return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &custom_op_forward, "custom forward (GPU)");
}
