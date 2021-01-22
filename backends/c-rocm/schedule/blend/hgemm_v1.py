# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ..standard.matmul_v1 import schedule as base_schedule

def schedule(attrs):
  base_schedule(attrs)

  attrs.blend = '''
#define dot2h1(x, y)  __hmul2(x, y)
#define dot2h2(x)     __hmul(((half*)&x)[0], ((half*)&x)[1])
'''
