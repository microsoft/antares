# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ..standard.matmul_v1 import schedule as base_schedule

def schedule(attrs):
  base_schedule(attrs)

  attrs.blend = '''
#define dot4a(x, y)   amd_mixed_dot(x, y, 0, false)
'''
