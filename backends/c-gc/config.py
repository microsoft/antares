# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import hashlib
import numpy as np

from antares.common import type_to_c as _native_dtype, AntaresGlobal

def get_execution_parallism():
  return 1

def do_native_translation_v2(codeset, **kwargs):
  if 'einstein_v2' not in kwargs['attrs'].ir:
    raise Exception("Program for graphcore must be based on Antares IR")

  kernel_name, args, body = codeset
  arg_bufs = AntaresGlobal.local_arg_pros

  func_args, delta_args = '', []
  for buf in arg_bufs['_in']:
    if buf['name'].startswith('_'):
      delta_args.append(buf['name'])
      continue
    func_args += ' Input<Vector<%s>> %s; // local size: %s\n' % (_native_dtype(buf['dtype']), buf['name'], buf['shape'])
  for buf in arg_bufs['_out']:
    func_args += ' Output<Vector<%s>> %s; // local size: %s\n' % (_native_dtype(buf['dtype']), buf['name'], buf['shape'])

  codelet_id = 'Vuid_%s' % hashlib.sha1(body.encode()).hexdigest()
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
