# Fused RoPE, FP8 Quantization, and Paged MLA Cache Append with AutoRT

This tutorial shows how to translate a PyTorch reference implementation into a fused CUDA/HIP kernel exported through AutoRT. The kernel:

1. Applies rotary position embeddings (RoPE) to query and key tensors.
2. Quantizes query and key data to FP8 with finite saturation.
3. Appends the quantized key data to paged MLA caches.
4. Skips padding tokens whose batch index is negative.

The implementation supports both split-half (NeoX) and interleaved RoPE layouts.
Unlike the FlashInfer fused operator shown below, the AutoRT implementation can
be compiled for either NVIDIA CUDA or AMD HIP/ROCm.

## Reference behavior

The operation receives these logical inputs:

- `q_rope`: query RoPE features with shape `[nnz, num_q_heads, rope_dim]`
- `k_rope`: key RoPE features with shape `[nnz, rope_dim]`
- `q_nope`: query non-RoPE features with shape `[nnz, num_q_heads, nope_dim]`
- `k_nope`: key non-RoPE features with shape `[nnz, nope_dim]`
- `cos_sin_cache`: cosine and sine values with shape `[max_position, rope_dim]`
- `pos_ids`: position IDs for all tokens
- `ckv_cache`: paged non-RoPE key cache
- `kpe_cache`: paged RoPE key cache
- `kv_indices` and `kv_indptr`: logical-to-physical page mapping
- `batch_indices` and `positions`: cache destination for each token

For valid tokens, the reference operation computes:

```python
q_rope_out = quantize_fp8(apply_rope(q_rope))
q_nope_out = quantize_fp8(q_nope)
kpe_cache[page, offset] = quantize_fp8(apply_rope(k_rope))
ckv_cache[page, offset] = quantize_fp8(k_nope)
```

Tokens with `batch_indices[token] < 0` are ignored.

## Equivalent PyTorch implementation

The following implementation expresses the complete operation using regular
PyTorch tensor operations. It is useful as a readable specification and as a
correctness reference for the fused kernels.

```python
import torch


def apply_rope_ref(
    x: torch.Tensor,
    pos_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> torch.Tensor:
    dim = x.shape[-1]
    half = dim // 2

    cache = cos_sin_cache[pos_ids.long()]
    cos = cache[..., :half]
    sin = cache[..., half:dim]

    # Add a broadcast dimension for every head axis.
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    x = x.float()

    if is_neox:
        # Split-half layout: [x1, x2] -> [-x2, x1].
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    else:
        # Interleaved layout: [x0, x1] -> [-x1, x0].
        rotated = torch.stack(
            (-x[..., 1::2], x[..., 0::2]),
            dim=-1,
        ).flatten(-2)
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

    return x * cos + rotated * sin


def quantize_fp8_ref(
    x: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    fp8_max = torch.finfo(dtype).max
    return (x.float() * scale).clamp(-fp8_max, fp8_max).to(dtype)


def rope_quantize_fp8_append_paged_mla_cache_ref(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool = True,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    page_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = ckv_cache.dtype
    q_rope_out = torch.empty_like(q_rope, dtype=fp8_dtype)
    q_nope_out = torch.empty_like(q_nope, dtype=fp8_dtype)

    # Negative batch indices identify padding tokens.
    token_idx = torch.nonzero(batch_indices >= 0, as_tuple=False).flatten()
    if token_idx.numel() == 0:
        return q_rope_out, q_nope_out

    token_pos_ids = pos_ids[token_idx]

    q_rope_rot = apply_rope_ref(
        q_rope[token_idx],
        token_pos_ids,
        cos_sin_cache,
        is_neox,
    )
    k_rope_rot = apply_rope_ref(
        k_rope[token_idx],
        token_pos_ids,
        cos_sin_cache,
        is_neox,
    )

    q_rope_out[token_idx] = quantize_fp8_ref(
        q_rope_rot,
        quant_scale_q,
        fp8_dtype,
    )
    q_nope_out[token_idx] = quantize_fp8_ref(
        q_nope[token_idx],
        quant_scale_q,
        fp8_dtype,
    )

    kpe = quantize_fp8_ref(
        k_rope_rot,
        quant_scale_kv,
        fp8_dtype,
    )
    ckv = quantize_fp8_ref(
        k_nope[token_idx],
        quant_scale_kv,
        fp8_dtype,
    )

    batch = batch_indices[token_idx].long()
    position = positions[token_idx].long()
    logical_page = kv_indptr[batch].long() + position // page_size
    physical_page = kv_indices[logical_page].long()
    page_offset = position % page_size

    kpe_cache[physical_page, page_offset] = kpe
    ckv_cache[physical_page, page_offset] = ckv

    return q_rope_out, q_nope_out
```

This implementation intentionally leaves output rows for padding tokens
uninitialized, matching the fused operation.

## Equivalent FlashInfer implementation

Recent FlashInfer versions provide the same fused MLA operation through
`flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache`. In MLA mode,
`k_rope` and `k_nope` are two-dimensional, `v` must be `None`, and the paged
cache tuple is `(ckv_cache, kpe_cache)`. This FlashInfer fused path uses its
CUDA implementation and is not available on ROCm.

```python
import torch
from flashinfer.rope import rope_quantize_fp8_append_paged_kv_cache


def rope_quantize_fp8_append_paged_mla_cache_flashinfer(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    k_nope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool = True,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    page_size: int = 16,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return rope_quantize_fp8_append_paged_kv_cache(
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        v=None,
        cos_sin_cache=cos_sin_cache,
        pos_ids=pos_ids,
        paged_kv_cache=(ckv_cache, kpe_cache),
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        batch_indices=batch_indices,
        positions=positions,
        is_neox=is_neox,
        quantize_dtype=ckv_cache.dtype,
        quant_scale_q=quant_scale_q,
        quant_scale_kv=quant_scale_kv,
        page_size=page_size,
        q_rope_out=None,
        q_nope_out=None,
        enable_pdl=enable_pdl,
    )
```

FlashInfer allocates and returns `(q_rope_out, q_nope_out)` and updates
`ckv_cache` and `kpe_cache` in place. It internally converts page metadata to
32-bit integers. The cosine/sine cache must be `torch.float32`, while the
source tensors must be `torch.float16` or `torch.bfloat16`.

## CUDA and ROCm portability

| Implementation | NVIDIA CUDA | AMD HIP/ROCm |
| --- | --- | --- |
| PyTorch reference | Yes, subject to PyTorch FP8 support | Subject to PyTorch and device FP8 support |
| FlashInfer fused operator shown above | Yes | No |
| AutoRT kernel in this tutorial | Yes | Yes |

The AutoRT kernel body uses CUDA/HIP-compatible C++17 constructs and AutoRT
tensor accessors. AutoRT selects the corresponding backend when installed in a
supported CUDA or ROCm environment. The documented AMD requirement is HIP
ROCm 7.2.4. The FlashInfer call in this tutorial dispatches to a CUDA-specific
fused kernel, so it cannot serve as the ROCm implementation of this operation.

## AutoRT design

AutoRT permits only one tensor after `->` in `@DEF_FUNC`. This operation logically returns both `q_rope_out` and `q_nope_out`, while also modifying two caches. Therefore:

- `q_rope_out` is the declared output.
- `q_nope_out` is preallocated and passed as a writable input.
- `ckv_cache` and `kpe_cache` are writable input tensors.

The exported signature below targets BF16 source tensors and FP8 output/cache tensors. Change the source annotations from `bfloat16` to `float16` or another supported type when necessary.

Each CUDA/HIP block handles one `(token, query_head)` pair. All blocks produce query outputs, while only the block for query head zero writes the shared key caches. Threads within a block stride across the feature dimensions.

## Export the kernel (Portable Kernel Generation based on Skills)

```sh
python3.12 -m autort.utils.export -n rope_quantize_fp8_append_paged_mla_cache --source='
@DEF_FUNC: q_rope:bfloat16[NNZ, QH, ROPE_DIM], k_rope:bfloat16[NNZ, ROPE_DIM], q_nope:bfloat16[NNZ, QH, NOPE_DIM], k_nope:bfloat16[NNZ, NOPE_DIM], cos_sin_cache:float32[MAX_POS, ROPE_DIM], pos_ids:int32[NNZ], ckv_cache:float8[NUM_PAGES, PAGE_SIZE, NOPE_DIM], kpe_cache:float8[NUM_PAGES, PAGE_SIZE, ROPE_DIM], kv_indices:int32[NUM_KV_INDICES], kv_indptr:int32[NUM_BATCH_PTRS], batch_indices:int32[NNZ], positions:int32[NNZ], q_nope_out:float8[NNZ, QH, NOPE_DIM] -> q_rope_out:float8[NNZ, QH, ROPE_DIM]
@DEF_BIND: ~NNZ~:1, ~QH~:1, ~%~:256
@DEF_EXTRA: is_neox:int32, quant_scale_q:float32, quant_scale_kv:float32, page_size:int32, fp8_max:float32

void main() {
  int nnz = int(size_of_NNZ());
  int q_heads = int(size_of_QH());
  int rope_dim = int(size_of_ROPE_DIM());
  int nope_dim = int(size_of_NOPE_DIM());
  int half = rope_dim / 2;

  __builtin_assume((rope_dim & 1) == 0);
  __builtin_assume(page_size > 0);

  int block = int(blockIdx.x);
  int token = block / q_heads;
  int head = block % q_heads;
  int tid = int(threadIdx.x);
  int stride = int(blockDim.x);

  if (token >= nnz || batch_indices(token) < 0)
    return;

  int pos_id = pos_ids(token);

  for (int d = tid; d < rope_dim; d += stride) {
    int trig_d = is_neox ? (d < half ? d : d - half) : d / 2;
    int rotated_d = is_neox
      ? (d < half ? d + half : d - half)
      : ((d & 1) ? d - 1 : d + 1);

    float32 x = float32(q_rope(token, head, d));
    float32 rotated = float32(q_rope(token, head, rotated_d));

    if ((is_neox && d < half) || (!is_neox && !(d & 1)))
      rotated = -rotated;

    float32 cos_v = cos_sin_cache(pos_id, trig_d);
    float32 sin_v = cos_sin_cache(pos_id, half + trig_d);
    float32 value = (x * cos_v + rotated * sin_v) * quant_scale_q;
    value = fminf(fp8_max, fmaxf(-fp8_max, value));
    q_rope_out(token, head, d) = (float8)value;
  }

  for (int d = tid; d < nope_dim; d += stride) {
    float32 value = float32(q_nope(token, head, d)) * quant_scale_q;
    value = fminf(fp8_max, fmaxf(-fp8_max, value));
    q_nope_out(token, head, d) = (float8)value;
  }

  if (head == 0) {
    int batch = batch_indices(token);
    int position = positions(token);
    int logical_page = kv_indptr(batch) + position / page_size;
    int physical_page = kv_indices(logical_page);
    int page_offset = position % page_size;

    for (int d = tid; d < rope_dim; d += stride) {
      int trig_d = is_neox ? (d < half ? d : d - half) : d / 2;
      int rotated_d = is_neox
        ? (d < half ? d + half : d - half)
        : ((d & 1) ? d - 1 : d + 1);

      float32 x = float32(k_rope(token, d));
      float32 rotated = float32(k_rope(token, rotated_d));

      if ((is_neox && d < half) || (!is_neox && !(d & 1)))
        rotated = -rotated;

      float32 cos_v = cos_sin_cache(pos_id, trig_d);
      float32 sin_v = cos_sin_cache(pos_id, half + trig_d);
      float32 value = (x * cos_v + rotated * sin_v) * quant_scale_kv;
      value = fminf(fp8_max, fmaxf(-fp8_max, value));
      kpe_cache(physical_page, page_offset, d) = (float8)value;
    }

    for (int d = tid; d < nope_dim; d += stride) {
      float32 value = float32(k_nope(token, d)) * quant_scale_kv;
      value = fminf(fp8_max, fmaxf(-fp8_max, value));
      ckv_cache(physical_page, page_offset, d) = (float8)value;
    }
  }
}'
```

## RoPE index mapping

The cosine/sine cache stores `rope_dim / 2` cosine values followed by the same number of sine values.

For split-half (NeoX) layout, the input is divided into two halves:

```text
[x1, x2] -> [-x2, x1]
```

The kernel maps output dimension `d` to:

```cpp
int trig_d = d < half ? d : d - half;
int rotated_d = d < half ? d + half : d - half;
```

The first half negates its rotated input.

For interleaved layout, adjacent elements form a pair:

```text
[x0, x1] -> [-x1, x0]
```

The kernel maps output dimension `d` to:

```cpp
int trig_d = d / 2;
int rotated_d = (d & 1) ? d - 1 : d + 1;
```

Even dimensions negate their rotated input.

Both layouts then evaluate:

```text
output = input * cosine + rotated_input * sine
```

## FP8 quantization

The PyTorch reference first converts values to FP32, applies a scale, clamps to the finite range of the destination FP8 type, and converts to FP8:

```python
(x.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
```

The kernel performs the same steps:

```cpp
float32 value = float32(x) * scale;
value = fminf(fp8_max, fmaxf(-fp8_max, value));
output = (float8)value;
```

`fp8_max` is supplied at runtime so the wrapper can obtain it from the actual cache dtype with `torch.finfo`.

## Call the exported operator

```python
import torch
import autort


def run_rope_quantize_fp8_append(
    q_rope,
    k_rope,
    q_nope,
    k_nope,
    cos_sin_cache,
    pos_ids,
    ckv_cache,
    kpe_cache,
    kv_indices,
    kv_indptr,
    batch_indices,
    positions,
    *,
    is_neox=True,
    quant_scale_q=1.0,
    quant_scale_kv=1.0,
    page_size=16,
):
    q_nope_out = torch.empty_like(q_nope, dtype=ckv_cache.dtype)

    q_rope_out = autort.ops.rope_quantize_fp8_append_paged_mla_cache(
        q_rope,
        k_rope,
        q_nope,
        k_nope,
        cos_sin_cache,
        pos_ids,
        ckv_cache,
        kpe_cache,
        kv_indices,
        kv_indptr,
        batch_indices,
        positions,
        q_nope_out,
        extra=[
            int(is_neox),
            float(quant_scale_q),
            float(quant_scale_kv),
            int(page_size),
            float(torch.finfo(ckv_cache.dtype).max),
        ],
    )

    return q_rope_out, q_nope_out
```

The call updates `ckv_cache` and `kpe_cache` in place. As in the reference implementation, output entries belonging to skipped padding tokens remain uninitialized.

## Profile the kernel

Use `autort.perform` to measure the exported operator:

```python
autort.perform(
    lambda: autort.ops.rope_quantize_fp8_append_paged_mla_cache(
        q_rope,
        k_rope,
        q_nope,
        k_nope,
        cos_sin_cache,
        pos_ids,
        ckv_cache,
        kpe_cache,
        kv_indices,
        kv_indptr,
        batch_indices,
        positions,
        q_nope_out,
        extra=[
            int(is_neox),
            float(quant_scale_q),
            float(quant_scale_kv),
            int(page_size),
            float(torch.finfo(ckv_cache.dtype).max),
        ],
    )
)
```

## Correctness requirements

- `rope_dim` must be even.
- `page_size` must be positive and compatible with the second cache dimension.
- `pos_ids` must index valid rows in `cos_sin_cache`.
- Valid tokens must reference valid batches, positions, logical pages, and physical pages.
- Cache destinations should be unique if deterministic writes are required.
- Tensor dtype annotations in `@DEF_FUNC` must match the tensors passed to the operator.
- If index tensors are stored as `int64`, update their `@DEF_FUNC` annotations accordingly.
