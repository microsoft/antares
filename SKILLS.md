## AutoRT System Requirement:

- OS: Ubuntu >= 24.04 or Windows >= 10;
- Python3 == 3.12.x;
- CUDA >= 12.0 for NVIDIA GPU, or HIP ROCm == 7.2.4 for AMD GPU;
- Setup:
```sh
python3.12 -m pip install --no-build-isolation -U https://github.com/microsoft/antares/releases/download/v0.9.6/autort-0.9.6.6.0+cuda.zip
```

## Agent SKILL:

- Please learn the format for writing CUDA/HIP kernels through the autort interface (PyTorch >= 2).
- In the code, there can be only one output after `->`. If you need multiple outputs, put one after `->` and put the rest on the input side.
- For extra writes, the user can pre-allocate sufficient workspace as input tensors and pass them as inputs before `->`.
- Outside `void main()`, NEVER define C++ macros, helper functions, structs, or headers. Their original prototypes / declarations must go inside the main body. For example, helper functions should be defined as lambdas inside the main body.
- To get the total dimension size of a tensor, e.g. `SPLIT` shown in the example below, please use: size_of_SPLIT(). Please note that size_of_XXX() is not a compile-time constant.
- The highest language standard allowed in the kernel is C++17.
- Limited to these tensor datatypes: float32/float64/float16/bfloat16/float8/int32/int64/int16/int8/..

## Below is an example:

```sh
python3.12 -m autort.utils.export -n cuda_kernel_example --source='
@DEF_FUNC: attn_logits:float32[B, SPLIT, H, M], attn_lse:float32[B, SPLIT, H], final_lse:float32[B, H] -> final_output:float32[B, H, M]
@DEF_BIND: ~B~:32, ~H~:8, ~%~:1024
@DEF_EXTRA: sm_scale:float32, kv_context:int32

void main() {
  // In DEF_BIND, it represents the implicit config information for kernel launch: cuda_kernel_example<<dim3(ceil(B / 32)) * ceil(H / 8), 1, 1), dim3(1024, 1, 1)>>>

  // Start to write the codes:

  // Example-1: Unpack index from merged blockIdx.x:
  int batch_id = int(blockIdx.x) / size_of_H();
  int head_id = int(blockIdx.x) % size_of_H();
  int thread_id = int(threadIdx.x);

  // Example-2: Define intra-block resources, e.g. LDS / ..
  __shared__ float32 local_attn_lse[8];

  // Example-3: Use tensor(A, B, ..) instead of tensor[A, B, ..] for all global inputs and outputs listed in @DEF_FUNC:
  local_attn_lse[0] = attn_lse(batch_id, 0, 0);
  __syncthreads();

  auto device_fn = []() -> float32 {
    return INFINITY;
  };

  // Example-4: But you are allowed for C++ style access by getting the starting pointer of global tensors, like data_ptr[A, B, ..].
  auto data_ptr = (float32* __restrict__)&attn_lse(0, 0, 0);
  data_ptr[0] = device_fn();

  
  // Example-5: "sm_scale, .." are non-constant scalar variables given from host on-the-fly, they are not necessary to show if no variables are required
  final_output(0, 1, 2) = sm_scale * sm_scale;

  // Example-6: You can print data if needed. HOWEVER, ensure using **single quote** for the outer quotes instead of `"` to avoid escape-sequence confusion used by printf internally.
  if (batch_id == 0 && thread_id == 0)
    printf("(autort) CUDA output = %g\n", final_output(0, 1, 2));
}'
```

## After compilation, here is the way for operator execution:
```py
import torch
import autort

device = autort.device()

#### Create kernels within Pytorch session:
# autort.export(name='cuda_kernel_example', source='''@DEF_FUNC: ...
#   ... ...
# ''')

attn_logits = torch.ones([2, 2, 2, 2], dtype=torch.float32).to(device)
attn_lse = torch.ones([2, 2, 2], dtype=torch.float32).to(device)
final_lse = torch.ones([2, 2], dtype=torch.float32).to(device)
final_output = autort.ops.cuda_kernel_example(attn_logits, attn_lse, final_lse, extra=[float(3.4), int(1234),])

print(final_output)
```

## For kernel profiling, please execute:
```py
import autort

autort.perform(lambda: autort.ops.cuda_kernel_example(attn_logits, attn_lse, final_lse, extra=[float(3.4), int(1234),]))
```
