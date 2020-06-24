# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ..standard.default import schedule as base_schedule

def schedule(attrs):
  base_schedule(attrs)

  attrs.blend = '''
#include <immintrin.h>

struct avx256{
    float value[8];
};

inline avx256 fastadd(const avx256& input0, const avx256& input1) {
  __m256 in0 = _mm256_loadu_ps(input0.value);
  __m256 in1 = _mm256_loadu_ps(input1.value);
  __m256 out = _mm256_add_ps(in0, in1);
  avx256 output;
  _mm256_storeu_ps(output.value, out);
  return output;
}
'''
