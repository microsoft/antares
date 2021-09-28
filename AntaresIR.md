### Antares IR Syntax

```
Einstein Expression Parsing:

  1) * Transform-based Operator Format:  output[D1, D2, ..] = f(input[D1, D2, ..]) where D2 in 8

       Step-1: Auto fill einsum dummy-axis, get:  output[D1, D2, ..] = f(input[D1, D2, ..]) where D2 in 8, D1 in input.shape[0], ..

       Step-2: Construct basic C code, get:

       for (int D1 = 0; D1 < input.shape[0]; D1++)
         for (int D2 = 0; D2 < 8; D2++)
           for (..)
             output[D1, D2, ..] = f(input[D1, D2, ..])

     * Specific Example 1 (OneHot Op): output0[N, F] = const(1.0).when([input0[N] == F], const(0.0)) where F in 128

       Step-1/Step-2: .., finally get:

       for (int N = 0; N < input0.shape[0]; ++N)
         for (int F = 0; F < 128; ++F)
           output0[N, F] = (input0[N] == F) ? const(1.0) : const(0.0);

     * Specific Example 2 (Arrange Op): output0[N] = N.cast(`float32`) where N in 1024

       Step-1/Step-2: .., finally get:

       for (int N = 0; N < 1024; ++N)
         output0[N] = static_cast<float>(N);


  2) * Aggregation-based Operator Format:  output[D1, D2, ..] +=! f(input[D1, D2, .., R1, R2, ..])

       Step-1: Auto fill einsum dummy-axis, get:  output[D1, D2, ..] +=! f(input[D1, D2, .., R1, R2, ..]) where D1 in input.shape[0], D2 in input.shape[1], .., R1 in input.shape[..], R2 in input.shape[..], ..

       Step-2: Construct basic C code, get:

       for (int D1 = 0; D1 < input.shape[0]; D1++)
         for (int D2 = 0; D2 < input.shape[1]; D2++)
           for (..)
             output[D1, D2, ..] = 0;
       for (int D1 = 0; D1 < input.shape[0]; D1++)
         for (int D2 = 0; D2 < input.shape[1]; D2++)
           for (..)
             for (int R1 = 0; R1 < input.shape[0]; R1++)
               for (int R2 = 0; R2 < input.shape[1]; R2++)
                 for (..)
                   output[D1, D2, ..] += f(input[D1, D2, .., R1, R2, ..]);

     * Specific Example (Conv2D Op): output0[N, F, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[F, C, KH, KW] where HO in 30, WO in 30

       Step-1/Step-2: .., finally get:

       for (int N = 0; N < input0.shape[0]; ++N)
         for (int F = 0; F < input1.shape[0]; ++F)
           for (int HO = 0; HO < 30; ++HO)
             for (int WO = 0; WO < 30; ++WO)
               output0[N, F, HO, WO] = 0;
       for (int N = 0; N < input0.shape[0]; ++N)
         for (int F = 0; F < input1.shape[0]; ++F)
           for (int HO = 0; HO < 30; ++HO)
             for (int WO = 0; WO < 30; ++WO)
               for (int C = 0; C < input1.shape[1]; ++C)         // R1
                 for (int KH = 0; KH < input1.shape[2]; ++KH)    // R2
                   for (int KW = 0; KW < input1.shape[3]; ++KW)  // R3
                     output0[N, F, HO, WO] += input0[N, C, HO + KH, WO + KW] * input1[F, C, KH, KW];

```
### Antares Built-in Primitive Mapping:
| Primitive Type | Antares IR Format | C Code Format |
|---|---|---|
| Branch (All) | ```x.when([c1, c2, ..], y)``` | ```(c1 && c2 && ..) ? x : y``` |
| Branch (Any) | ```x.when([c1, c2, ..], y, merge_op=`any`)``` | ```(c1 \|\| c2 \|\| ..) ? x : y``` |
| Type Cast | ```x.cast(`int8`)``` | ```static_cast<char>(x)``` |
| Function Call (Single Arg) | ```x.call(`exp`)``` | ```exp(x)``` |
| Function Call (Multiple Args) | ```x.call(`max`, [y, ..])``` | ```max(x, y, ..)``` |
| Logical Ops | ```x & ~(y \| z)``` | ```x && !(y \|\| z)``` |

### Antares Built-in Data Type Mapping:
| Antares Type | C Type |
|---|---|
| float64 | double |
| float32 | float |
| float16 | half |
| int32 | int |
| int16 | short |
| int8 | char |

### Antares Built-in Functions:
| Function Name | Proto | Explanation |
|---|---|---|
| max | T max(T, T) | The max value of two inputs |
| min | T min(T, T) | The min value of two inputs |
| log | T log(T) | The log value of input |
| exp | T exp(T) | The exponentiation of input |
| sqrt | T sqrt(T) | The square root of input |
| pow | T pow(T, T) | The power value of inputs: a ^ b |
| floor | int32/int64 floor(float32/float64) | The floor integer of input |
| ceil | int32/int64 ceil(float32/float64) | The ceil integer of input |
| remainder | float32/float64 remainder(float32/float64) | The remainder float of input |


### Detailed Examples:
```sh
# Broadcast
COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] = input0[N] where F in 32, HO in 2, WO in 2", input_dict={"input0": {"dtype": "float32", "shape": [16]}})' make

# BroadcastAll
COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] = input0[0] where N in 8, F in 32, HO in 2, WO in 2", input_dict={"input0": {"dtype": "float32", "shape": [1]}})' make

# MatMul
COMPUTE_V1='- einstein_v2("output0[N, M] +=! input0[N, K] * input1[K, M]", { "input0": {"dtype": "float32", "shape": [1024, 512]}, "input1": {"dtype": "float32", "shape": [512, 512]}})' make

# MatMulBiasAdd
COMPUTE_V1='- einstein_v2("output0[N, M] +=! input0[N, K] * input1[K, M] + input2[M] / K.val()", { "input0": {"dtype": "float32", "shape": [1024, 512]}, "input1": {"dtype": "float32", "shape": [512, 512]}, "input2": {"dtype": "float32", "shape": [512]} })' make

# BatchMatMul
COMPUTE_V1='- einstein_v2("output0[B, N, M] +=! input0[B, N, K] * input1[B, K, M]", input_dict={"input0": {"dtype": "float32", "shape": [3, 1024, 512]}, "input1": {"dtype": "float32", "shape": [3, 512, 512]}})' make

# Elementwise
COMPUTE_V1='- einstein_v2("output0[N] = input0[N] + input1[N]", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}})' make

# Scaler Compute
COMPUTE_V1='- einstein_v2("output0[] = input0[] + input1[]", input_dict={"input0": {"dtype": "float32", "shape": []}, "input1": {"dtype": "float32", "shape": []}})' make

# Multiple Outputs (in same shape)
COMPUTE_V1='- einstein_v2("output0[N] = input0[N] + input1[N]; output1[N] = input0[N] * 2; output2[N] = input1[N] + output1[N];", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}}, extra_outputs=["output0", "output1", "output2"])' make

# Transpose
COMPUTE_V1='- einstein_v2("output0[N, C, H, W] = input0[N, H, W, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 229, 229, 3]}})' make

# Reshape
COMPUTE_V1='- einstein_v2("output0[A, B, C] = input0[A, B, C / 64, C % 64] where C in 128", input_dict={"input0": {"dtype": "float32", "shape": [3, 3, 2, 64]}})' make

# ReduceSum
COMPUTE_V1='- einstein_v2("output0[N] +=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 1024]}})' make

# ReduceMin
COMPUTE_V1='- einstein_v2("output0[N] <=! input0[N, C]", input_dict={"input0": {"dtype": "float32", "shape": [32, 1024]}})' make

# Cast
COMPUTE_V1='- einstein_v2("output0[N] = N.cast(`float32`) where N in 1024", {})' make

# Condition Relu
COMPUTE_V1='- einstein_v2("output0[N, C] = input0[N, C].when([input0[N, C] > 0.0], 0.0)", input_dict={"input0": {"dtype": "float32", "shape": [1024, 512]}})' make

# Condition Relu for dynamtic data type
COMPUTE_V1='- einstein_v2("output0[N, C] = input0[N, C].when([input0[N, C] > const(0.0).cast(input0[N, C].dtype())], const(0.0).cast(input0[N, C].dtype()))", input_dict={"input0": {"dtype": "float32", "shape": [1024, 512]}})' make

# `Range + Tanh` using External Function
COMPUTE_V1='- einstein_v2("output0[N] = N.cast(`float32`).call(`tanh`) where N in 1024", {})' make

# ConvolutionNoPad
COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[F, C, KH, KW] where HO in 30, WO in 30", { "input0": {"dtype": "float32", "shape": [16, 64, 32, 32]}, "input1": {"dtype": "float32", "shape": [256, 64, 3, 3]}})' make

# ConvolutionWithPad
COMPUTE_V1='- _N, _CI, _H, _W, _CO, _KH, _KW, _SH, _SW, _PH, _PW = 16, 64, 32, 32, 256, 3, 3, 1, 1, 0, 0; _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; einstein_v2(f"output0[N, F, HO, WO] +=! input0[N, C, HO * {_SH} + KH - {_PH}, WO * {_SW} + KW - {_PW}].when([HO * {_SH} + KH - {_PH} >= 0, HO * {_SH} + KH - {_PH} < {_H}, WO * {_SW} + KW - {_PW} >= 0, WO * {_SW} + KW - {_PW} < {_W}], 0.0) * input1[F, C, KH, KW] where HO in {_HO}, WO in {_WO}", { "input0": {"dtype": "float32", "shape": [_N, _CI, _H, _W]}, "input1": {"dtype": "float32", "shape": [_CO, _CI, _KH, _KW]}})' make

# ConvolutionWithPad (Fused reduce axis)
COMPUTE_V1='- _N, _CI, _H, _W, _CO, _KH, _KW, _SH, _SW, _PH, _PW = 16, 64, 32, 32, 256, 3, 3, 1, 1, 0, 0; \
  _HO, _WO = (_H - _KH + _PH * 2) // _SH + 1, (_W - _KW + _PW * 2) // _SW + 1; \
  einstein_v2(f" \
    output0[N, F, HO, WO] +=! input0[N, CKHKW // {_KW * _KH}, HO * {_SH} + ((CKHKW % {_KW * _KH}) // {_KW}) - {_PH}, WO * {_SW} + ((CKHKW % {_KW * _KH}) % {_KW}) - {_PW}].when([HO * {_SH} + ((CKHKW % {_KW * _KH}) // {_KW}) - {_PH} >= 0, HO * {_SH} + ((CKHKW % {_KW * _KH}) // {_KW}) - {_PH} < {_H}, WO * {_SW} + ((CKHKW % {_KW * _KH}) % {_KW}) - {_PW} >= 0, WO * {_SW} + ((CKHKW % {_KW * _KH}) % {_KW}) - {_PW} < {_W}], 0.0) * input1[F, CKHKW] where HO in {_HO}, WO in {_WO} \
  ", { "input0": {"dtype": "float32", "shape": [_N, _CI, _H, _W]}, "input1": {"dtype": "float32", "shape": [_CO, _CI * _KH * _KW]}})' make

# ConvWinograd_3x3 (_KH = _KW = 3, _SH = _SW = 1, _PH = _PW = 0)
COMPUTE_V1='- _N, _CI, _H, _W, _CO = 16, 64, 32, 32, 256; _HO, _WO = _H - 2, _W - 2; _nH, _nW = (_HO + 1) // 2, (_WO + 1) // 2; _P = _N * _nH * _nW; einstein_v2(f"helper4x3[N, M] = const(1.0).when([N * 3 + M == 0, N * 3 + M == 11], const(0.0).when([N * 3 + M == 1, N * 3 + M == 2, N * 3 + M == 9, N * 3 + M == 10], const(-0.5).when([N * 3 + M == 4], 0.5, merge_op=`any`), merge_op=`any`), merge_op=`any`) where N in 4, M in 3; transform_filter[EPS, NU, CI, CO] +=! ((input1[CO, CI, Rkh, Rkw] * helper4x3[EPS, Rkh] * helper4x3[NU, Rkw])); input_tile[C, B, EPS, NU] = input0[B // ({_nH} * {_nW}), C, B // {_nW} % {_nH} * 2 + EPS, B % {_nW} * 2 + NU] where C in {_CI}, B in {_P}, EPS in 4, NU in 4; helper4x4[N, M] = const(1.0).when([N * 4 + M == 0, N * 4 + M == 6, N * 4 + M == 9, N * 4 + M == 10, N * 4 + M == 15], const(-1.0).when([N * 4 + M == 5, N * 4 + M == 7, N * 4 + M == 8], 0.0, merge_op=`any`), merge_op=`any`) where N in 4, M in 4; transform_input[EPS, NU, C, B] +=! input_tile[C, B, K1, K2] * helper4x4[K1, EPS] * helper4x4[K2, NU] where EPS in 4, NU in 4, C in {_CI}, B in {_P}; batch_gemm[EPS, NU, K, B] +=! transform_filter[EPS, NU, CI, K] * transform_input[EPS, NU, CI, B] where EPS in 4, NU in 4, K in {_CO}, B in {_P}; helper4x2[N, M] = const(0.0) .when([N * 2 + M == 1, N * 2 + M == 6], const(-1.0).when([N * 2 + M == 3], 1.0, merge_op=`any`), merge_op=`any`) where N in 4, M in 2; inverse[K, B, VH, VW] +=! batch_gemm[K1, K2, K, B] * helper4x2[K1, VH] * helper4x2[K2, VW] where K in {_CO}, B in {_P}, VH in 2, VW in 2; output0[N, K, H, W] = inverse[K, N * {_nH} * {_nW} + H // 2 * {_nW} + W // 2, H % 2, W % 2] where N in {_N}, K in {_CO}, H in {_HO}, W in {_WO}", {"input0": {"dtype": "float32", "shape": [_N, _CI, _H, _W]}, "input1": {"dtype": "float32", "shape": [_CO, _CI, 3, 3]}})' make

# ConvWinograd_3x3 with external helper matrix
COMPUTE_V1='- _N, _CI, _H, _W, _CO = 16, 64, 32, 32, 256; _HO, _WO = _H - 2, _W - 2; _nH, _nW = (_HO + 1) // 2, (_WO + 1) // 2; _P = _N * _nH * _nW; einstein_v2(f""" \
  transform_filter[EPS, NU, CI, CO] +=! ((input1[CO, CI, Rkh, Rkw] * helper4x3[EPS, Rkh] * helper4x3[NU, Rkw])); \
  input_tile[C, B, EPS, NU] = input0[B // ({_nH} * {_nW}), C, B // {_nW} % {_nH} * 2 + EPS, B % {_nW} * 2 + NU] where C in {_CI}, B in {_P}, EPS in 4, NU in 4; \
  transform_input[EPS, NU, C, B] +=! input_tile[C, B, K1, K2] * helper4x4[K1, EPS] * helper4x4[K2, NU] where EPS in 4, NU in 4, C in {_CI}, B in {_P}; \
  batch_gemm[EPS, NU, K, B] +=! transform_filter[EPS, NU, CI, K] * transform_input[EPS, NU, CI, B] where EPS in 4, NU in 4, K in {_CO}, B in {_P}; \
  inverse[K, B, VH, VW] +=! batch_gemm[K1, K2, K, B] * helper4x2[K1, VH] * helper4x2[K2, VW] where K in {_CO}, B in {_P}, VH in 2, VW in 2; \
  output0[N, K, H, W] = inverse[K, N * {_nH} * {_nW} + H // 2 * {_nW} + W // 2, H % 2, W % 2] where N in {_N}, K in {_CO}, H in {_HO}, W in {_WO} \
""", {"input0": {"dtype": "float32", "shape": [_N, _CI, _H, _W]}, "input1": {"dtype": "float32", "shape": [_CO, _CI, 3, 3]}, "helper4x2": {"dtype": "float32", "shape": [4, 2]}, "helper4x3": {"dtype": "float32", "shape": [4, 3]}, "helper4x4": {"dtype": "float32", "shape": [4, 4]}})' make


# DepthToSpace
COMPUTE_V1='- einstein_v2("output0[N, H, C0, W, C1, C2] = input0[N, H, W, C0, C1, C2]", input_dict={"input0": {"dtype": "float32", "shape": [1, 256, 256, 2, 2, 4]}})' make

# DepthwiseConv
COMPUTE_V1='- einstein_v2("output0[N, C, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[KH, KW, C, 0] where HO in 30, WO in 30", input_dict={"input0": {"dtype": "float32", "shape": [32, 16, 32, 32]}, "input1": {"dtype": "float32", "shape": [3, 3, 16, 1]}})' make

# Slice
COMPUTE_V1='- einstein_v2("output0[N, F] = input0[N, F, 2]", input_dict={"input0": {"dtype": "float32", "shape": [1, 16, 32]}})' make

# Concat
COMPUTE_V1='- einstein_v2("output0[N, F] = input0[N, F].when([F < 128], input1[N, F - 128]) where F in 256", input_dict={"input0": {"dtype": "float32", "shape": [4, 128]}, "input1": {"dtype": "float32", "shape": [4, 128]}})' make

# OneHot
COMPUTE_V1='- einstein_v2("output0[N, F] = const(1.0).when([input0[N] == F], const(0.0)) where F in 128", input_dict={"input0": {"dtype": "int32", "shape": [4]}})' make

# Take
COMPUTE_V1='- einstein_v2("output0[F, C] = input0[input1[F], C]", input_dict={"input0": {"dtype": "float32", "shape": [30528, 1024]}, "input1": {"dtype": "int32", "shape": [3072]}})' make

# Gather
COMPUTE_V1='- einstein_v2("output0[N, F] = input0[input1[N, F]]", input_dict={"input0": {"dtype": "float32", "shape": [65536]}, "input1": {"dtype": "int32", "shape": [4, 64]}})' make

# Pad
COMPUTE_V1='- einstein_v2("output0[N, C, HO, WO] = input0[N, C, -1 + HO, -1 + WO].when([-1 + HO >= 0, -1 + HO < 32, -1 + WO >= 0, -1 + WO < 32], 0.0) where HO in 34, WO in 34", input_dict={"input0": {"dtype": "float32", "shape": [32, 3, 32, 32]}})' make

# DivNoNan
COMPUTE_V1='- einstein_v2("output0[N] = (input0[N] / input1[N]).when([input1[N] != 0], 0.0)", input_dict={"input0": {"dtype": "float32", "shape": [32 * 1024]}, "input1": {"dtype": "float32", "shape": [32 * 1024]}})' make

# MaxPool
COMPUTE_V1='- einstein_v2("output0[N, C, HO, WO] >=! input0[N, C, HO * 2 + KH, WO * 2 + KW] where HO in 6, WO in 6, KW in 2, KH in 2", input_dict={"input0": {"dtype": "float32", "shape": [32, 3, 12, 12]}})' make

# AvgPool
COMPUTE_V1='- einstein_v2("output0[NC, HO, WO] +=! input0[NC, HO * 3 + KH, WO * 3 + KW] / 9.0 where HO in 85, WO in 85, KW in 3, KH in 3", input_dict={"input0": {"dtype": "float32", "shape": [1024, 255, 255]}})' make

# Tile
COMPUTE_V1='- einstein_v2("output0[ON, OC] = input0[ON % 2, OC % 16] where ON in 1024, OC in 4096", input_dict={"input0": {"dtype": "float32", "shape": [2, 16]}})' make

# Softmax
COMPUTE_V1='- einstein_v2("temp0[N] >=! input0[N, C]; temp1[N] +=! (input0[N, C] - temp0[N]).call(`exp`); output0[N, C] = (input0[N, C] - temp0[N]).call(`exp`) / temp1[N]", { "input0": {"dtype": "float32", "shape": [32, 1024]} })' make

# BatchNorm Inference
COMPUTE_V1='- einstein_v2("output0[N, C, H, W] = bias[C] + scale[C] * (input0[N, C, H, W] - mean[C]) / (1e-5 + variance[C]).call(`sqrt`)", input_dict={"input0": {"dtype": "float32", "shape": [16, 256, 16, 16]}, "mean": {"dtype": "float32", "shape": [256]}, "variance": {"dtype": "float32", "shape": [256]}, "scale": {"dtype": "float32", "shape": [256]}, "bias": {"dtype": "float32", "shape": [256]} })' make

# Logical Bool Operation
COMPUTE_V1='- einstein_v2("output0[N, M] = input0[N, M] & ~input1[N, M]", { "input0": {"dtype": "int8", "shape": [1024, 512]}, "input1": {"dtype": "int8", "shape": [1024, 512]} })' make

# Sigmoid
COMPUTE_V1='- einstein_v2("output0[N, M] = 1.0 / (1.0 + (-input0[N, M]).call(`exp`))", { "input0": {"dtype": "float32", "shape": [1024, 512]} })' make

# AddMatMul Head Fusion
COMPUTE_V1='- einstein_v2("temp0[K, N] = input0[N, K] + 100; output0[N, M] +=! temp0[K, N] * input1[K, M] where K in 10", { "input0": {"dtype": "float32", "shape": [1024, 512]}, "input1": {"dtype": "float32", "shape": [512, 512]}})' make

# ConvBiasRelu Tail Fusion
COMPUTE_V1='- einstein_v2("conv_out[N, F, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[KH, KW, C, F] where HO in 256, WO in 256; conv_bias[N, F, HO, WO] = conv_out[N, F, HO, WO] + input2[0, 0, 0, F]; output0[N, F, HO, WO] = conv_bias[N, F, HO, WO].when(conv_bias[N, F, HO, WO] > 0.0, 0.0)", input_dict={"input0": {"dtype": "float32", "shape": [1, 16, 256, 256]}, "input1": {"dtype": "float32", "shape": [1, 1, 16, 16]}, "input2": {"dtype": "float32", "shape": [1, 1, 1, 16]}})' make

# Scatter4D
COMPUTE_V1='- _B, _M = 2, 8; einstein_v2("data[indices[B, 0], indices[B, 1], indices[B, 2], indices[B, 3], M] =. updates[B, M]", input_dict={"data": {"dtype": "float32", "shape": [32, 32, 32, 32, _M]}, "indices": {"dtype": "int32", "shape": [_B, 4]}, "updates": {"dtype": "float32", "shape": [_B, _M]}})' make
