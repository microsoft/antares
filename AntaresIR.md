### Antares IR Syntax

```sh
# Broadcast
COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] = input0[N] where F in 32, HO in 2, WO in 2", input_dict={"input0": {"dtype": "float32", "shape": [16]}})' make

# BroadcastAll
COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] = input0[0] where N in 8, F in 32, HO in 2, WO in 2", input_dict={"input0": {"dtype": "float32", "shape": [1]}})' make

# MatMul
COMPUTE_V1='- einstein_v2("output0[N, M] +=! input0[N, K] * input1[K, M]", { "input0": {"dtype": "float32", "shape": [1024, 512]}, "input1": {"dtype": "float32", "shape": [512, 512]}})' make

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

# External Function
COMPUTE_V1='- einstein_v2("output0[N] = N.cast(`float32`).call(`tanh`) where N in 1024", {})' make

# ConvolutionNoPad
COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[F, C, KH, KW] where HO in 30, WO in 30", { "input0": {"dtype": "float32", "shape": [16, 64, 32, 32]}, "input1": {"dtype": "float32", "shape": [256, 64, 3, 3]}})' make

# ConvolutionWithPad
COMPUTE_V1='- _N, _C, _HW, _F, _K, _S, _P = 2, 64, 27, 192, 5, 1, 2; _HWO = (_HW - _K + _P * 2) // _S + 1; einstein_v2("output0[N, F, HO, WO] +=! input0[N, C, HO * %d + KH - %d, WO * %d + KW - %d].when([HO * %d + KH - %d >= 0, HO * %d + KH - %d < %d, WO * %d + KW - %d >= 0, WO * %d + KW - %d < %d], 0.0) * input1[F, C, KH, KW] where HO in %d, WO in %d" % (_S, _P, _S, _P, _S, _P, _S, _P, _HW, _S, _P, _S, _P, _HW, _HWO, _HWO), { "input0": {"dtype": "float32", "shape": [_N, _C, _HW, _HW]}, "input1": {"dtype": "float32", "shape": [_F, _C, _K, _K]}})' make

# DepthToSpace
COMPUTE_V1='- einstein_v2("output0[N, H, C0, W, C1, C2] = input0[N, H, W, C0, C1, C2]", input_dict={"input0": {"dtype": "float32", "shape": [1, 256, 256, 2, 2, 4]}})' make

# DepthwiseConv
COMPUTE_V1='- einstein_v2("output0[N, C, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[C, 0, KH, KW] where HO in 30, WO in 30", input_dict={"input0": {"dtype": "float32", "shape": [32, 16, 32, 32]}, "input1": {"dtype": "float32", "shape": [16, 1, 3, 3]}})' make

# Slice
COMPUTE_V1='- einstein_v2("output0[N, F] = input0[N, F, 2]", input_dict={"input0": {"dtype": "float32", "shape": [1, 16, 32]}})' make

# Concat
COMPUTE_V1='- einstein_v2("output0[N, F] = input0[N, F].when([F < 128], input1[N, F - 128]) where F in 256", input_dict={"input0": {"dtype": "float32", "shape": [4, 128]}, "input1": {"dtype": "float32", "shape": [4, 128]}})' make

# OneHot
COMPUTE_V1='- einstein_v2("output0[N, F] = const(1.0).when([input0[N] == F], 0.0) where F in 128", input_dict={"input0": {"dtype": "int32", "shape": [4]}})' make

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
COMPUTE_V1='- einstein_v2("temp0[NC, HO, WO] +=! input0[NC, HO * 3 + KH, WO * 3 + KW] where HO in 85, WO in 85, KW in 3, KH in 3; output0[NC, HO, WO] = temp0[NC, HO, WO] * 0.111111", input_dict={"input0": {"dtype": "float32", "shape": [1024, 255, 255]}})' make

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

####### Experimental (Platform-dependent)
# MatMul (explicit plan)
COMPUTE_V1='- einstein_v2("output0[N, M] +=! input0[N, K] * input1[K, M]", { "input0": {"dtype": "float32", "shape": [1024, 512]}, "input1": {"dtype": "float32", "shape": [512, 512]}})  ## @: plan/matmul_v1' make

# ReduceSum (backend-related explicit plan combinations)
COMPUTE_V1='- einstein_v2("output0[N] +=! input0[N, M]", {"input0": {"dtype": "float32", "shape": [1024, 3072]}})  ## @: plan/c-cuda=reduce_sum_v1,c-rocm=reduce_sum_v1,default' make

# AddMatMul Head Fusion
COMPUTE_V1='- einstein_v2("temp0[K, N] = input0[N, K] + 100; output0[N, M] +=! temp0[K, N] * input1[K, M] where K in 10", { "input0": {"dtype": "float32", "shape": [1024, 512]}, "input1": {"dtype": "float32", "shape": [512, 512]}})' make

# ConvBiasRelu Tail Fusion
COMPUTE_V1='- einstein_v2("conv_out[N, F, HO, WO] +=! input0[N, C, HO + KH, WO + KW] * input1[KH, KW, C, F] where HO in 256, WO in 256; conv_bias[N, F, HO, WO] = conv_out[N, F, HO, WO] + input2[0, 0, 0, F]; output0[N, F, HO, WO] = conv_bias[N, F, HO, WO].when(conv_bias[N, F, HO, WO] > 0.0, 0.0)", input_dict={"input0": {"dtype": "float32", "shape": [1, 16, 256, 256]}, "input1": {"dtype": "float32", "shape": [1, 1, 16, 16]}, "input2": {"dtype": "float32", "shape": [1, 1, 1, 16]}})  ## @: plan/convfwd_nchw_v1' make

# [INTRISIC SPEC] ROCm/CUDA's Custom Argmax2D
BACKEND=c-rocm COMPUTE_V1='- einstein_v2("output0[N] argmax(0, N)=! input0[N, C].call(`index_of`, [N], dtype=`int32`)", input_dict={"input0": {"dtype": "float32", "shape": [32, 128]}})  ## @: plan/c-cuda=blend.my_argmax,c-rocm=blend.my_argmax' make

# [INTRISIC SPEC] HLSL ElementwiseExp of VecFloat
BACKEND=c-hlsl COMPUTE_V1='- einstein_v2("output0[N] = input0[N].call(`my_exp`)", input_dict={"input0": {"dtype": "float4@128", "shape": [16]}})  ## @: plan/c-hlsl=blend.my_exp' make

# [INTRISIC SPEC] ROCm MatMul (mixed type for float16 -> float16)
BACKEND=c-rocm COMPUTE_V1='- einstein_v2("temp0[N, M] +=! input0[N, K].call(`dot2h1`, [input1[K, M]]); output0[N, M] = temp0[N, M].call(`dot2h2`, dtype=`float16`)", { "input0": {"dtype": "half2@32", "shape": [1024, 512]}, "input1": {"dtype": "half2@32", "shape": [512, 512]}})  ## @: plan/c-rocm=blend.hgemm_v1' make

# [INTRISIC SPEC] ROCm MatMul (mixed type for int8x4 -> int32)
BACKEND=c-rocm COMPUTE_V1='- einstein_v2("output0[N, M] +=! input0[N, K].call(`dot4a`, [input1[K, M]], dtype=`int32`)", { "input0": {"dtype": "int32", "shape": [1024, 512]}, "input1": {"dtype": "int32", "shape": [512, 512]}})  ## @: plan/c-rocm=blend.dot4a_v1' make

# [INTRISIC SPEC] CPU AVX Add (elementwise add using CPU AVX instructions)
BACKEND=c-mcpu COMPUTE_V1='- einstein_v2("output0[N] = input0[N].call(`fastadd`, [input1[N]])", input_dict={"input0": {"dtype": "avx256@256", "shape": [16]}, "input1": {"dtype": "avx256@256", "shape": [16]}})  ## @: plan/c-mcpu=blend.avx_add' make

# [INTRISIC SPEC] CUDA FP16 Tensorcore
BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, M] +=! input0[N, K].cast(`float32`) * input1[K, M].cast(`float32`)", { "input0": {"dtype": "float16", "shape": [1024, 1024]}, "input1": {"dtype": "float16", "shape": [1024, 1024]}})  ## @: plan/c-cuda=blend.matmul_fp16_tensorcore|layout=NN' make
