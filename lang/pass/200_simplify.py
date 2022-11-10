# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import itertools
from common import backend, AntaresGlobal
from lang.einstein_v2 import walk_in_ast, OpTensor


def no_trivial_ax_input(ast_seq, global_input_dict, global_output_dict):
  for i, ast in enumerate(ast_seq):
    ax_elim = []
    for ax in ast['props']['reduce_axes'] + ast['props']['data_axes']:
      if ax['range'] == 1:
        ax_elim.append(ax['name'])
    ast['props']['reduce_axes'] = [x for x in ast['props']['reduce_axes'] if x['name'] not in ax_elim]
    if len(ast_seq) == 1 and len(global_output_dict) == 1:
      ax_rebuld = [x for x in ast['props']['data_axes'] if x['name'] not in ax_elim]
      if ax_rebuld:
        ast['props']['data_axes'] = ax_rebuld
      else:
        ast['props']['data_axes'] = [ast['props']['data_axes'][0]]
    if not ast['props']['reduce_axes']:
      ast['props']['reduce_type'] = None

    def scan_trivial_axis(root, ancestor, ax_elim):
      if root._op == 'axis' and root._value in ax_elim:
        return OpTensor('const', 0, 'int32')
    walk_in_ast(ast, 'root', scan_trivial_axis, [ax_elim])

def update_global_dict(ast_seq, global_input_dict, global_output_dict):
  for ast in ast_seq:
    for k in ast['props']['input_dict']:
      if k in global_input_dict:
        global_input_dict[k] = ast['props']['input_dict'][k]
    k = ast['props']['output_name']
    if k in global_output_dict:
      global_output_dict[k] = {"shape": [x['range'] for x in ast['props']['data_axes']], "dtype": ast['root']._dtype}

def run_pass_v2(ast_seq, global_input_dict, global_output_dict):
  return
  # Just a rough check
  if int(os.environ.get('TVM', 1)) == 0:
    return
  if 'plan/' in os.environ.get('COMPUTE_V1', ''):
    return
  if os.environ.get('NO_SIMPLIFY', '') or backend in ('c-ipu',):
    return
  no_trivial_ax_input(ast_seq, global_input_dict, global_output_dict)
  update_global_dict(ast_seq, global_input_dict, global_output_dict)

