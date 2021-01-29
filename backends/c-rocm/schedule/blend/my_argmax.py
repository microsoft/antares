# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from ..standard.default import schedule as base_schedule

def schedule(attrs):
  base_schedule(attrs)

  input_name = attrs.inputs[0].op.name

  input_shape = attrs.inputs[0].shape
  assert len(input_shape) == 2, "This Argmax scheduler for 2D input tensor only."

  attrs.blend = '''
#define argmax(idx, idy, batch) \
  ({0}[batch * {1} + idx] > {0}[batch * {1} + idy] ? idx : idy)
#define index_of(in, batch) \
  (&in - {0} - batch * {1})
'''.format(input_name, input_shape[1])
