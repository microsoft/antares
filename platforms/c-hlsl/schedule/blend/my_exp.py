# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ..standard.default import schedule as base_schedule

def schedule(attrs):
  base_schedule(attrs)

  attrs.blend = '''
float4 my_exp(float4 val) {
  return exp(val);
}
'''
