# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
import copy
import json
import numpy as np

# Tensor name: the first charactor must be lower case letter, and the following charactors must be within [a-zA-Z_]
# Axis name: the first charactor must be upper case letter, and the following charactors must be within [a-zA-Z]

full_tensor_dict = None
explicit_range = None

class OpTensor:
    @staticmethod
    def parse(other, output_dtype=None):
        if isinstance(other, OpTensor):
          return other.cast(output_dtype)
        if output_dtype is not None:
          return OpTensor('const', other, output_dtype)
        if isinstance(other, int):
          return OpTensor('const', other, 'int32')
        if isinstance(other, float):
          return OpTensor('const', other, 'float32')
        raise Exception("Unrecognized const node type: %s" % type(other))

    @staticmethod
    def merge_dtype(first, second):
        dtypes = (first._dtype, second._dtype)
        ordered_dtypes = ['float64', 'float32', 'int32', 'int16', 'int8']
        for _dtype in ordered_dtypes:
          if _dtype in dtypes:
            return _dtype
        return first._dtype

    def dtype(self):
        return self._dtype

    def __init__(self, _op, _value, _dtype):
        self._op = _op
        self._value = _value
        self._dtype = _dtype

    def __repr__(self):
        return 'OpTensor{"%s", "%s", "%s"}' % (self._op, self._value, self._dtype)

    def __getitem__(self, key):
        if self._op != 'tensor':
            raise Exception("The instance to access its dim values must be a tensor array.")
        key = list(key if isinstance(key, tuple) else (key, ))
        for i in range(len(key)):
          key[i] = OpTensor.parse(key[i])
          it = key[i]
          if it._op == 'axis' and explicit_range[it._value] is None:
            explicit_range[it._value] = full_tensor_dict[self._value]['shape'][i]
        return OpTensor('get_item', {"tensor": self, "index": key}, self._dtype)

    # Calculation Ops
    def __mul__(self, other):
        other = OpTensor.parse(other)
        output_dtype = OpTensor.merge_dtype(self, other)
        if other._op == 'const' and other._value == 1:
            return self.cast(output_dtype)
        if self._op == 'const' and self._value == 1:
            return other.cast(output_dtype)
        return OpTensor('op', {"name": "*", "inputs": [self.cast(output_dtype), other.cast(output_dtype)]}, output_dtype)

    def __rmul__(self, other):
        other = OpTensor.parse(other)
        return other.__mul__(self)

    def __truediv__(self, other):
        other = OpTensor.parse(other)
        op_name = '//' if self._dtype == 'int32' and other._dtype == 'int32' else '/'
        output_dtype = OpTensor.merge_dtype(self, other)
        if other._op == 'const' and other._value == 1:
            return self.cast(output_dtype)
        if other._op == 'const' and self._op == 'axis':
            assert self._value in explicit_range and explicit_range[self._value] is not None
            if op_name == '//' and explicit_range[self._value] < other._value:
                return OpTensor.parse(0, output_dtype)
        return OpTensor('op', {"name": op_name, "inputs": [self, other]}, output_dtype)

    def __rtruediv__(self, other):
        other = OpTensor.parse(other)
        return other.__truediv__(self)

    def __floordiv__(self, other):
        other = OpTensor.parse(other)
        return self.__truediv__(other)

    def __rfloordiv__(self, other):
        other = OpTensor.parse(other)
        return other.__floordiv__(self)

    def __mod__(self, other):
        other = OpTensor.parse(other)
        if other._op == 'const':
            assert other._dtype == 'int32'
            if other._value == 1:
                return OpTensor.parse(0, self._dtype)
            if self._op == 'axis':
                assert self._value in explicit_range and explicit_range[self._value] is not None
                if explicit_range[self._value] <= other._value:
                    return self
        return OpTensor('op', {"name": "%", "inputs": [self, other]}, self._dtype)

    def __add__(self, other):
        other = OpTensor.parse(other)
        output_dtype = OpTensor.merge_dtype(self, other)
        if other._op == 'const' and other._value == 0:
            return self.cast(output_dtype)
        if self._op == 'const' and self._value == 0:
            return other.cast(output_dtype)
        return OpTensor('op', {"name": "+", "inputs": [self.cast(output_dtype), other.cast(output_dtype)]}, output_dtype)

    def __radd__(self, other):
        other = OpTensor.parse(other)
        return other.__add__(self)

    def __sub__(self, other):
        other = OpTensor.parse(other)
        output_dtype = OpTensor.merge_dtype(self, other)
        if other._op == 'const' and other._value == 0:
            return self.cast(output_dtype)
        return OpTensor('op', {"name": "-", "inputs": [self.cast(output_dtype), other.cast(output_dtype)]}, output_dtype)

    def __rsub__(self, other):
        other = OpTensor.parse(other)
        return other.__sub__(self)

    def __neg__(self):
        return OpTensor.parse(0, self._dtype).__sub__(self)

    # Relation Ops
    def __lt__ (self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "<", "inputs": [self, other]}, 'int8')

    def __le__ (self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "<=", "inputs": [self, other]}, 'int8')

    def __gt__ (self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "<", "inputs": [other, self]}, 'int8')

    def __ge__ (self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "<=", "inputs": [other, self]}, 'int8')

    def __eq__ (self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "==", "inputs": [self, other]}, 'int8')

    def __ne__ (self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "!=", "inputs": [self, other]}, 'int8')

    def __and__(self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "&", "inputs": [self, other]}, 'int8')

    def __or__(self, other):
        other = OpTensor.parse(other)
        return OpTensor('op', {"name": "|", "inputs": [self, other]}, 'int8')

    def __invert__(self):
        return OpTensor('op', {"name": "~", "inputs": [self]}, 'int8')

    # Special Ops
    def cast(self, output_dtype):
        if output_dtype is None or self._dtype == output_dtype:
          return self
        return OpTensor('cast', {"inputs": [self]}, output_dtype)

    def call(self, func_name, others=None, output_dtype=None):
        if others is None:
          others = []
        for i in range(len(others)):
          others[i] = OpTensor.parse(others[i])
        if output_dtype is None:
          output_dtype = self._dtype
        return OpTensor('call', {"name": func_name, "inputs": [self] + others}, output_dtype)

    def when(self, conditions, other, merge_op='all'):
        other = OpTensor.parse(other)
        assert self._dtype == other._dtype or '@' in self._dtype or '@' in other._dtype, "Conditional true and false values must have same datatype (%s v.s. %s)" % (self._dtype, other._dtype)
        conditions = conditions if isinstance(conditions, list) else [conditions]
        for cond in conditions:
          assert cond._dtype == 'int8', 'Each condition in when statement must be boolean(int8) type, get: %s' % cond._dtype
        return OpTensor('when', {"if": conditions, "true": self, "false": other, "merge_op": merge_op}, self._dtype)

def parse_to_ast(expr):
  global full_tensor_dict
  expr = expr.strip().replace('`', '"').replace('\'', '"')
  if re.search('\[ *\]', expr):
    expr = re.sub('\[ *\]', '[0]', expr)
    if expr.rfind('where') == -1:
      expr += ' where Scaler in 1'
    else:
      expr += ', Scaler in 1'
  at_index = expr.rfind(' where ')
  if at_index != -1:
    range_desc = expr[at_index + len(' where '):]
    expr = expr[:at_index]
  else:
    range_desc = ''

  # Parse compute axes & init axis nodes
  global explicit_range
  explicit_range = {}
  brac_st = False
  for i in range(len(expr)):
    if expr[i] == '"':
      brac_st = not brac_st
      continue
    if i < 1 or brac_st:
      continue
    if expr[i].isupper() and (not expr[i - 1].isalpha()) and (not expr[i - 1].isdigit()) and (expr[i - 1] != '_'):
      for j in range(i, len(expr) + 1):
        if j == len(expr) or (not expr[j].isalpha() and not expr[j].isdigit()):
            ax_name = expr[i:j]
            break
      if ax_name not in explicit_range:
        explicit_range[ax_name] = None
  exec("_id = OpTensor('axis', '_id', 'int32')")
  for k in explicit_range:
    exec("%s = OpTensor('axis', k, 'int32')" % k)

  # Parse where clause
  for x in range_desc.split(','):
    x = x.strip()
    if not x:
      continue
    k, v = x.split(' in ')
    explicit_range[k.strip()] = int(v.strip())

  # Parse compute set-op, get lval & rval
  props = {'data_axes': [], 'reduce_axes': [], 'input_dict': None, 'output_name': None, 'reduce_type': None}

  at_index = expr.find('=')
  if expr[at_index - 1] != ' ':
    if expr[at_index - 1] in ('<', '>', '+'):
      props['reduce_type'] = expr[at_index - 1]
      lval = expr[:at_index - 1].strip()
    else:
      blank_index = expr.find(' ', 0, at_index)
      assert blank_index > 0, "Illegal reduce naming in equation near: `L-value <reduce_type>=`"
      props['reduce_type'] = expr[blank_index + 1:at_index]
      lval = expr[:blank_index].strip()
  else:
    lval = expr[:at_index].strip()
  if expr[at_index + 1] == '!':
    assert(props['reduce_type'] is not None)
    rval = expr[at_index + 2:].strip()
  else:
    rval = expr[at_index + 1:].strip()

  # Distinguish data/reduce axes according to lval
  for x in lval[lval.index('[') + 1:lval.rindex(']')].split(','):
    x = x.strip()
    if x == '0':
      x = 'Scaler'
    props['data_axes'].append(x)
  for x in explicit_range:
    if x not in props['data_axes']:
      props['reduce_axes'].append(x)

  for input_name in full_tensor_dict:
    if not input_name[0].islower():
      raise Exception("Tensor variable name must start with lower case letter: %s" % input_name)
    exec('%s = OpTensor("tensor", input_name, "%s")' % (input_name, full_tensor_dict[input_name]["dtype"]))
    
  # Build ast according to rval & fill uncertain axis range
  _root = eval(rval)
  for x in explicit_range:
    if explicit_range[x] is None:
      raise Exception("The range of axis `%s` is undeterminzed, please use `where` clause to set the range explicitly." % x)

  # Collect output inferences
  props['data_axes'] = [{'name': x, 'range': explicit_range[x]} for x in props['data_axes']]
  props['reduce_axes'] = [{'name': x, 'range': explicit_range[x]} for x in props['reduce_axes']]

  output_name = lval[:lval.index('[')].strip()
  props['output_name'] = output_name

  ast = {'props': props, 'root': _root}

  input_names = set()
  def scan_items(root, input_names):
    if root._op != 'get_item':
      return
    input_names.add(root._value['tensor']._value)
  walk_in_ast(ast['root'], scan_items, [input_names,], ast, 'root')

  local_input_dict = {}
  for name in input_names:
    local_input_dict[name] = full_tensor_dict[name]
  props['input_dict'] = local_input_dict
  return ast

def const(other):
  return OpTensor.parse(other)

def warp_axis(ax_name):
  assert(ax_name[0].isupper() or ax_name == '_id')
  return ax_name

def emit_antares_ir(ast):
  def _emit(node):
    if node._op == 'const':
      return 'const(%s)' % node._value
    elif node._op == 'axis':
      if hasattr(node, '_func'):
        return node._func(node._value)
      return node._value
    elif node._op == 'op':
      if len(node._value['inputs']) == 2:
        return '(%s %s %s)' % (_emit(node._value['inputs'][0]), node._value['name'], _emit(node._value['inputs'][1]))
      raise
    elif node._op == 'get_item':
      return '%s[%s]' % (node._value['tensor']._value, ', '.join([_emit(x) for x in node._value['index']]))
    elif node._op == 'call':
      if len(node._value['inputs']) == 1:
        return '(%s).call(`%s`, dtype=`%s`)' % (_emit(node._value['inputs'][0]), node._value['name'], node._dtype)
      return '(%s).call(`%s`, [%s], dtype=`%s`)' % (_emit(node._value['inputs'][0]), node._value['name'], ', '.join([_emit(x) for x in node._value['inputs'][1:]]), node._dtype)
    elif node._op == 'when':
      if len(node._value['if']) == 0:
        return '(%s)' % _emit(node._value['true'])
      return '(%s).when([%s], %s, merge_op="%s")' % (_emit(node._value['true']), ', '.join([_emit(x) for x in node._value['if']]), _emit(node._value['false']), node._value['merge_op'])
    elif node._op == 'cast':
      return '(%s).cast(`%s`)' % (_emit(node._value['inputs'][0]), node._dtype)
    else:
      raise Exception("Emit Antares IR: Unhanled reverse-emit op type: %s" % node._op)
  lval = '%s[%s]' % (ast['props']['output_name'], ', '.join([x['name'] for x in ast['props']['data_axes']]))
  comp_type = '%s=%s' % (ast['props']['reduce_type'] if ast['props']['reduce_type'] else '', '!' if ast['props']['reduce_type'] else '')
  return '%s %s %s where %s;' % (lval, comp_type, _emit(ast['root']), ', '.join(['%s in %d' % (x['name'], x['range']) for x in ast['props']['data_axes'] + ast['props']['reduce_axes']]))

def emit_tvm_body(node, props):
  if node._op == 'const':
    return 'tir.const(%s, dtype="%s")' % (node._value, node._dtype)
  elif node._op == 'get_item':
    tensor = node._value['tensor']
    index = node._value['index']
    _str = tensor._value + '['
    if len(index) > 0:
      for i, it in enumerate(index):
        _str += emit_tvm_body(it, props) + ', '
      _str = _str[:-2] + ']'
    return _str
  elif node._op == 'axis':
    axis_name = warp_axis(node._value)
    if hasattr(node, '_func'):
      axis_name = node._func(axis_name)
    return axis_name
  elif node._op == 'op':
    op_name = node._value["name"]
    op_input_size = len(node._value["inputs"])
    if op_name in ('&', '|', '~'):
      if op_name == '&':
        return 'te.all(' + emit_tvm_body(node._value["inputs"][0], props) + '.astype("bool"), ' + emit_tvm_body(node._value["inputs"][1], props) + '.astype("bool"))'
      elif op_name == '|':
        return 'te.any(' + emit_tvm_body(node._value["inputs"][0], props) + '.astype("bool"), ' + emit_tvm_body(node._value["inputs"][1], props) + '.astype("bool"))'
      else:
        return '(' + emit_tvm_body(node._value["inputs"][0], props) + ' == 0)'
    elif op_input_size == 2:
      return '(' + emit_tvm_body(node._value["inputs"][0], props) + ' ' + op_name + ' ' + emit_tvm_body(node._value["inputs"][1], props) + ')'
    elif op_input_size == 1:
      return '(' + op_name + emit_tvm_body(node._value["inputs"][0], props) + ')'
    else:
      raise Exception('Unrecognized op type: %s[%d]' % (op_name, op_input_size))
  elif node._op == 'cast':
    return '%s.astype(cast_dtype("%s"))' % (emit_tvm_body(node._value["inputs"][0], props), node._dtype)
  elif node._op == 'call':
    return 'tir.call_pure_extern(cast_dtype("%s"), "%s", %s)' % (node._dtype, node._value['name'], ', '.join([emit_tvm_body(x, props) for x in node._value["inputs"]]))
  elif node._op == 'when':
    all_conds = [emit_tvm_body(cond, props) for cond in node._value['if']]
    return 'tir.if_then_else(te.%s(' % node._value['merge_op'] + ', '.join(all_conds) + '), t=' + emit_tvm_body(node._value['true'], props) + ', f=' + emit_tvm_body(node._value['false'], props) + ')'
  else:
    raise Exception('Unrecognized node type: %s' % node._op)

def walk_in_ast(node, func, args, parent, attr_id):

  def _walk(node, parent, attr_id):
    updated_node = func(node, *args)
    if updated_node is not None:
      if isinstance(updated_node, str) and updated_node == '':
        return
      updated_node = copy.deepcopy(updated_node)
      if isinstance(parent, OpTensor):
        setattr(parent, attr_id, updated_node)
      else:
        parent[attr_id] = updated_node
      return
    if node._op == 'get_item':
      for i, ch in enumerate(node._value['index']):
        _walk(ch, node._value['index'], i)
    elif node._op in ['op', 'call', 'cast']:
      for i, ch in enumerate(node._value['inputs']):
        _walk(ch, node._value['inputs'], i)
    elif node._op == 'when':
      for i, ch in enumerate(node._value['if']):
        _walk(ch, node._value['if'], i)
      _walk(node._value['true'], node._value, 'true')
      _walk(node._value['false'], node._value, 'false')
    elif node._op in ['axis', 'const']:
      pass
    else:
      raise Exception('Unhandled node type in walk_in_ast(): %s' % node._op)

  _walk(node, parent, attr_id)

def ir_graph_parser(exprss, input_dict, extra_outputs):
  statements = [s_.strip() for s_ in exprss.split(';')]
  global full_tensor_dict
  full_tensor_dict = copy.deepcopy(input_dict)
  output_dict = {}
  ast_seq = []
  for s in statements:
    if not s:
      continue
    ast = parse_to_ast(s)
    k = ast['props']['output_name']
    ast_outputs_dict = {k: {"shape": [x['range'] for x in ast['props']['data_axes']], "dtype": ast['root']._dtype}}
    full_tensor_dict[k] = ast_outputs_dict[k]
    if k in extra_outputs:
      output_dict[k] = ast_outputs_dict[k]
    ast_seq.append(ast)
  os.environ['MEDIATE_TENSORS'] = json.dumps(full_tensor_dict)

  # Also include the last output
  if k not in extra_outputs:
    output_dict[k] = ast_outputs_dict[k]

  # Registry Global Argument Properties
  arg_props = {'_in': [], '_out': []}
  for k in input_dict:
    prop = copy.deepcopy(input_dict[k])
    prop['name'] = k
    arg_props['_in'].append(prop)
  for k in output_dict:
    prop = copy.deepcopy(output_dict[k])
    prop['name'] = k
    arg_props['_out'].append(prop)
  arg_props['_in'].sort(key=lambda x: x['name'])
  arg_props['_out'].sort(key=lambda x: x['name'])

  from antares.common import AntaresGlobal
  AntaresGlobal.global_arg_pros = arg_props

  import importlib
  passes = os.listdir('lang/pass')
  passes.sort()
  for pas in passes:
    if pas.endswith('.py'):
      pass_stage = importlib.import_module('lang.pass.%s' % pas[:-3])
      pass_stage.run_pass_v2(ast_seq, input_dict, output_dict)

  # Generate LL_IR body for ast_seq
  def emit_input_body(input_dict):
    input_body = '_id = input("_id", [1], dtype="int32")[0]; '
    for key in input_dict:
      input_info = input_dict[key]
      input_body += '%s = input("%s", %s, dtype="%s"); ' % (key, key, input_info['shape'], input_info['dtype'])
    return input_body

  def emit_reduce_body(ast):
    reduce_body, reduce_set = '', []
    props = ast['props']
    if props['reduce_axes']:
      for x in props['reduce_axes']:
        axis_name = warp_axis(x['name'])
        reduce_set.append(axis_name)
        reduce_body += '%s = loop(%d, name="%s"); ' % (axis_name, x['range'], axis_name)
      reduce_maps = {'+': 'te.sum', '>': 'te.max', '<': 'te.min'}
      if props['reduce_type'] in reduce_maps:
        reduce_func = reduce_maps[props['reduce_type']]
      else:
        spec_idx = props['reduce_type'].find('(')
        if spec_idx >= 0:
          reduce_func = 'common_reduce("%s", %s)' % (props['reduce_type'][:spec_idx], props['reduce_type'][spec_idx:])
        else:
          reduce_func = 'common_reduce("%s")' % props['reduce_type']
      reduce_pattern = '%s(' % reduce_func + '%s' + ', axis=[%s])' % ', '.join(reduce_set)
    else:
      reduce_pattern = '%s'
    return reduce_body, reduce_pattern

  def emit_output_body(ast, reduce_pattern):
    root, props = ast['root'], ast['props']
    output_shape = [x['range'] for x in props['data_axes']]
    output_name = props['output_name']
    output_begin = '%s = output(shape=%s, func=lambda %s: ' % (output_name, output_shape, ', '.join([warp_axis(x['name']) for x in props['data_axes']]))
    basic_body = emit_tvm_body(root, props)
    output_end = ', dtype="%s", tag="%s", name="%s", final_output=%s); ' % (root._dtype, '', output_name, output_name in output_dict)
    return output_begin + reduce_pattern % basic_body + output_end

  ll_irs = [emit_input_body(input_dict)]
  for ast in ast_seq:
    loops_def, pattern = emit_reduce_body(ast)
    ll_irs.append(loops_def + emit_output_body(ast, pattern))
  return '\n'.join(ll_irs)
