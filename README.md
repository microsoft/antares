## AutoRT: the Next Generation of Antares.

***AutoRT for Device Runtime:***

AutoRT is a compiler solution that helps runtime users to invent, benchmark and optimize operators for Pytorch using your own accelerators.

#### Recommended Docker Image for AutoRT Setup:

 * **Docker Image for CUDA:** nvidia/cuda:12.8.0-devel-ubuntu24.04
 * **Docker Image for ROCM:** rocm/dev-ubuntu-24.04:7.2.4
 * ..

## Tutorial of AutoRT:

```md
>>> Agentic Step-1:
Please read this SKILL and learn by yourself:
https://raw.githubusercontent.com/microsoft/antares/refs/heads/latest/SKILLS.md

>>> Agentic Step-2:
Regardless of whether the current environment meets the requirements,
please follow AutoRT kernel examples to write a similar command that computes: output_1d = torch.sum(input_2d, dim=-1)
All syntax for kernel body follows CUDA / HIP C++ 17 standard. No need for deep API consult.
```

## Quick Start:

```sh
autort -n hello_world_example --source='
@DEF_FUNC: input:float32[B, M] -> output:float32[B, M]
@DEF_BIND: ~B~:1, ~%~:1024

void main() {
  if (int(blockIdx.x) == 0 && int(threadIdx.x) == 0)
    printf("(autort) Hello World!\n");
}'
```

For Earlier Manual Doc (deprecated), please check: [Manual](README-manual.md) and [Legacy](README-legacy.md).

See More Information about Microsoft [Contributing](CONTRIBUTING.md) and [Trademarks](TRADEMARKS.md).
