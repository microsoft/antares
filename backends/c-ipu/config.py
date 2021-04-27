# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import hashlib
import numpy as np


def get_execution_parallism():
  return 1

def do_native_translation_v2(codeset, **kwargs):
  if 'einstein_v2' not in kwargs['attrs'].ir:
    raise Exception("Program for graphcore must be based on Antares IR")

  kernel_name, in_args, out_args, body = codeset

  func_args, delta_args = '', []
  for buf in in_args:
    if buf[1].startswith('_'):
      delta_args.append(buf[1])
      continue
    func_args += ' Input<Vector<%s>> %s;\n' % (buf[0], buf[1])
  for buf in out_args:
    func_args += ' Output<Vector<%s>> %s;\n' % (buf[0], buf[1])

  blend_code = kwargs['attrs'].blend.strip()
  blend_code = 'namespace {\n%s\n}\n\n' if blend_code else ''

  from antares.common import local_get_dir_file
  try:
    with open(local_get_dir_file('range_book.json'), 'r') as fp:
      range_book = json.load(fp)
  except FileNotFoundError:
    raise Exception("TODO: Graphcore body generation is not completely implemented in new emit_tvm_ir_v2()")

  props = []
  for k in range_book['book']:
    arr2d = range_book['book'][k]
    arr2d = [str(x)[1:-1].replace(', ', ',') for x in arr2d]
    arr2d = '/'.join(arr2d)
    props.append(k + '/' + arr2d)
  props = ';'.join(props)

  full_body = f'''// Antares Property (k * ax_id + l .. r): {props}

#include <poplar/Vertex.hpp>

using namespace poplar;

{blend_code}
class CODELET_{kernel_name}: public Vertex {{
public:
 bool compute() {{
  {body}
  return true;
 }}

{func_args}}};
'''
  return full_body
