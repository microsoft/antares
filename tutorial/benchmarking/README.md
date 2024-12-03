## Tutorial - Benchmark your Device:

#### Quick Test 1: Evaluate device memory bandwidth for CUDA/ROCm/DirectX 12/..
```sh
$ python -m autort.benchmark.memtest
  ...
  [1000/1000] AutoRT Device Memory Bandwidth: (Actual ~= 468.12 GB/s) (Theoretical ~= 561.75 GB/s)
```

#### Quick Test 2: Evaluate device FP32 performance for CUDA/ROCm/DirectX 12/..
```sh
$ python -m autort.benchmark.fp32test
  ...
  [5000/5000] AutoRT FP32 TFLOPS: (Actual ~= 9.84 TFLOPS) (Theoretical ~= 10.93 TFLOPS)
```

#### Quick Test 3: Evaluate device GEMM performance for CUDA/ROCm/DirectX 12/..
```sh
$ python -m autort.benchmark.mmtest
  ...
   `MM-Perf` (current) = 13.85 TFLOPS, 9.92 msec.
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ...
```

#### Quick Test 4: Evaluate host-device copy bandwidth.
```sh
$ python -m autort.benchmark.copytest
  ...
  HtoD(API) Bandwidth: 11.046 GB/s
  PtoD(API) Bandwidth: 12.209 GB/s
  DtoH(API) Bandwidth: 13.167 GB/s
  HtoH(API) Bandwidth: 18.779 GB/s
  DtoD(API) Bandwidth: 535.212 GB/s
```

#### Quick Test 5: Evaluate device kernel launching latency.
```sh
$ python -m autort.benchmark.launchtest
  ...
  [5000/5000] Device Launch Overhead Test: 1.781 us / kerne
  ...
```

