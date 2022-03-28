# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os
import copy
import importlib
import json
import re

custom_dtypes = {"@": 0}

def codegen(ast_seq, input_dict, output_dict, best_config, space_only=False):
  # Generate tvm body for ast_seq

  def warp_axis(ax_name):
    assert(ax_name[0].isupper() or ax_name == '_id')
    return ax_name

  def cast_dtype(dtype):
    idx = dtype.find('@')
    if idx < 0:
      return dtype

    # Update register list
    global custom_dtypes
    typename = dtype[:idx]
    if typename not in custom_dtypes:
      if len(custom_dtypes) <= 4:
        dtype_code = 150 + len(custom_dtypes)
      else:
        oldest = min(filter(lambda x: x != '@', custom_dtypes.keys()), key=lambda x: custom_dtypes[x][1])
        dtype_code = custom_dtypes[oldest][0]
        custom_dtypes.pop(oldest)
    else:
      dtype_code = custom_dtypes[typename][0]

    custom_dtypes[typename] = (dtype_code, custom_dtypes["@"], dtype)
    custom_dtypes["@"] += 1
    target.datatype.register(typename, dtype_code)

    bits = int(dtype[idx + 1:])
    if bits % 32 == 0:
      return 'custom[%s]32' % (dtype[:idx])
    else:
      raise Exception("Not support custom dtype of bits = %d" % bits)

  def emit_tvm_body(node, props):
    if node._op == 'const':
      return 'tvm.tir.const(%s, dtype="%s")' % (node._value, node._dtype)
    elif node._op == 'axis_range':
      return 'tvm.tir.const(%s, dtype="%s")' % (props['explicit_range'][node._value], node._dtype)
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
          return 'tvm.te.all(' + emit_tvm_body(node._value["inputs"][0], props) + '.astype("bool"), ' + emit_tvm_body(node._value["inputs"][1], props) + '.astype("bool"))'
        elif op_name == '|':
          return 'tvm.te.any(' + emit_tvm_body(node._value["inputs"][0], props) + '.astype("bool"), ' + emit_tvm_body(node._value["inputs"][1], props) + '.astype("bool"))'
        else:
          return '(' + emit_tvm_body(node._value["inputs"][0], props) + ' == 0)'
      elif op_name == '//':
        return 'tvm.tir.truncdiv(' + emit_tvm_body(node._value["inputs"][0], props) + ', ' + emit_tvm_body(node._value["inputs"][1], props) + ')'
      elif op_input_size == 2:
        return '(' + emit_tvm_body(node._value["inputs"][0], props) + ' ' + op_name + ' ' + emit_tvm_body(node._value["inputs"][1], props) + ')'
      elif op_input_size == 1:
        return '(' + op_name + emit_tvm_body(node._value["inputs"][0], props) + ')'
      else:
        raise Exception('Unrecognized op type: %s[%d]' % (op_name, op_input_size))
    elif node._op == 'cast':
      return '%s.astype("%s")' % (emit_tvm_body(node._value["inputs"][0], props), cast_dtype(node._dtype))
    elif node._op == 'call':
      return 'tvm.tir.call_pure_extern("%s", "%s", %s)' % (cast_dtype(node._dtype), node._value['name'], ', '.join([emit_tvm_body(x, props) for x in node._value["inputs"]]))
    elif node._op == 'when':
      all_conds = [emit_tvm_body(cond, props) for cond in node._value['if']]
      return 'tvm.tir.if_then_else(tvm.te.%s(' % node._value['merge_op'] + ', '.join(all_conds) + '), t=' + emit_tvm_body(node._value['true'], props) + ', f=' + emit_tvm_body(node._value['false'], props) + ')'
    else:
      raise Exception('Unrecognized node type: %s' % node._op)

  def emit_input_body(input_dict):
    input_body = ['_id = tvm.te.placeholder([1], dtype="int32", name="_id")[0]']
    for key in input_dict:
      input_info = input_dict[key]
      input_body += ['%s = tvm.te.placeholder(%s, dtype="%s", name="%s")' % (key, input_info['shape'], cast_dtype(input_info['dtype']), key)]
    return input_body

  def emit_reduce_body(ast):
    reduce_body, reduce_set = '', []
    props = ast['props']
    if props['reduce_axes']:
      for x in props['reduce_axes']:
        axis_name = warp_axis(x['name'])
        reduce_set.append(axis_name)
        reduce_body += '%s = tvm.te.reduce_axis((0, %d), name="%s")\n' % (axis_name, x['range'], axis_name)
      reduce_maps = {'+': 'tvm.te.sum', '>': 'tvm.te.max', '<': 'tvm.te.min'}
      if props['reduce_type'] in reduce_maps:
        reduce_func = reduce_maps[props['reduce_type']]
      else:
        reduce_func = 'tvm.te.comm_reducer(lambda x, y: tvm.tir.call_pure_extern(x.dtype, "%s", x, y), lambda min_t: tvm.tir.const(0, dtype=min_t), name="%s")' % (props['reduce_type'], props['reduce_type'])
      reduce_pattern = '%s(' % reduce_func + '%s' + ', axis=[%s])' % ', '.join(reduce_set)
    else:
      reduce_pattern = '%s'
    return reduce_body, reduce_pattern

  def emit_output_body(ast, reduce_pattern):
    root, props = ast['root'], ast['props']
    output_shape = [x['range'] for x in props['data_axes']]
    output_name = props['output_name']
    output_begin = '%s = tvm.te.compute(%s, lambda %s: (' % (output_name, output_shape, ', '.join([warp_axis(x['name']) for x in props['data_axes']]))
    basic_body = emit_tvm_body(root, props)
    output_end = ').astype("%s"), tag="", name="%s")\n' % (cast_dtype(root._dtype), output_name)
    return output_begin + reduce_pattern % basic_body + output_end

  ll_irs = ['import tvm'] + emit_input_body(input_dict)
  for ast in ast_seq:
    loops_def, pattern = emit_reduce_body(ast)
    ll_irs.append(loops_def + emit_output_body(ast, pattern))
  ll_irs = '\n'.join(ll_irs)

  sandbox = dict()
  exec(ll_irs, sandbox)

  from antares.common import Mock, AutoConfig, AntaresGlobal, product, backend
  if not hasattr(AntaresGlobal, 'auto_config'):
    AntaresGlobal.auto_config = AutoConfig()

  def emit_codegen():
    tvm = sandbox['tvm']
    inputs = sorted([sandbox[x] for x in input_dict], key=lambda x: x.name)
    outputs = sorted([sandbox[x] for x in output_dict], key=lambda x: x.op.name)

    program = os.environ['COMPUTE_V1'].strip()
    anno, options = program.find('## @'), []
    if anno >= 0:
      program, options = program[:anno].strip(), program[program.index(':', anno) + 1:].strip().split('|')

    if len(outputs) > 1:
      def to_list(shape):
        return [int(d) for d in shape]
      for i in range(1, len(outputs)):
        assert to_list(outputs[0].shape) == to_list(outputs[i].shape), "Shape sizes for multiple outputs should be equal: %s v.s. %s" % (to_list(outputs[0].shape), to_list(outputs[i].shape))
      outputs = tvm.te.compute(outputs[0].shape, lambda *X: [v[X] for v in outputs], name='MultipleOutputsTempVar')
    sch = tvm.te.create_schedule([outputs[i].op for i in range(len(outputs))])

    def get_device_props():
      props = tvm.runtime.ndarray.gpu(0)
      with open('%s/device_properties.cfg' % os.environ['ANTARES_DRIVER_PATH'], 'r') as fp:
        mem_bandwith = []
        while True:
          line = fp.readline()
          if not line:
            break
          key, val = line.split(': ')
          if key in ('GlobalMemoryBusWidth', 'MemoryClockRate'):
            mem_bandwith.append(float(val))
        mem_bandwith = 'inf' if not mem_bandwith else product(mem_bandwith) * 2.5e-7
        props.mem_bandwith = float(mem_bandwith)
      return props

    def do_native_scheduling(attrs):
      def select_plan(plan_name):
        if plan_name.find('.') < 0:
          plan_name = 'standard.' + plan_name
        schedule_lib = importlib.import_module('backends.%s.schedule.%s' % (attrs.backend, plan_name), __loader__.name)
        schedule_lib.schedule(attrs)

      plan = 'default'
      for opt in attrs.options:
        if opt.startswith('plan/'):
          for plan_name in opt[5:].split(','):
            idx = plan_name.find('=')
            if idx >= 0:
              backend, name = plan_name.split('=')
              plan = None
              if backend != attrs.backend:
                continue
              plan = name
            else:
              plan = plan_name
            break
      if plan is None:
        raise Exception("No available plan configured for backend: %s" % attrs.backend)
      try:
        return select_plan(plan)
      except ModuleNotFoundError:
        traceback.print_exc()
        return None

    def _callback(explicit_ops):
      attrs = Mock()
      attrs.device_props = get_device_props()
      attrs.inputs = list(inputs)
      attrs.outputs = list(outputs)
      attrs.explicit_ops = explicit_ops
      attrs.scheduler = sch
      attrs.backend = backend
      attrs.ir = program
      attrs.options = options
      attrs.blend = ''
      attrs.auto_config = AntaresGlobal.auto_config
      attrs.get_extent = lambda axis: int(axis.dom.extent)

      def get_lower():
        return str(tvm.lower(sch, attrs.inputs + attrs.outputs, simple_mode=True)).split('#[metadata]')[0]

      attrs.get_lower = get_lower
      AntaresGlobal.attrs = attrs
      do_native_scheduling(attrs)

    def traverse_inline(s, final_op, callback):
      visited = set()
      explicit_ops = []

      def _traverse(op):
          if op in visited:
              return
          visited.add(op)
          for tensor in op.input_tensors:
            if isinstance(tensor.op, tvm.te.tensor.ComputeOp):
              _traverse(tensor.op)
          if op.reduce_axis:
            explicit_ops.append(op)
          elif op not in s.outputs:
            s[op].compute_inline()
          else:
            explicit_ops.append(op)

      _traverse(final_op)
      callback(explicit_ops)

    traverse_inline(sch, outputs[0].op, _callback)
    return sch, AntaresGlobal.attrs.inputs + AntaresGlobal.attrs.outputs

  best_config = best_config if isinstance(best_config, dict) else json.loads(best_config)
  AntaresGlobal.auto_config.set_candidate(best_config)

  tvm, tvm_target = sandbox['tvm'], 'cuda'
  with tvm.target.Target(tvm_target):
    s, arg_bufs = emit_codegen()

  if space_only:
    return

  '''
  lower_source = str(tvm.lower(s, arg_bufs, simple_mode=True))
  lower_file = local_get_dir_file('my_kernel.lower', dir_sid=dir_sid)
  with open(lower_file, 'w') as fp:
    fp.write(lower_source)
  '''

  func = tvm.build(s, arg_bufs, tvm_target, name='template_op')
  assert(len(func.imported_modules) == 1)

  def verify_body(kernel_name, body):
    max_threads_per_block = AntaresGlobal.attrs.device_props.max_threads_per_block
    max_shared_memory_per_block = AntaresGlobal.attrs.device_props.max_shared_memory_per_block
    assert max_threads_per_block > 0 and max_shared_memory_per_block >= 0, '[Error] Invalid device properties, maybe device is not detected correctly.'

    thread_extents, shared_mem_in_bytes = dict(), 0
    thread_extent_symbol, shared_symbol = '// [thread_extent] ', '__shared__ '
    for line in body.split('\n'):
      ll = line.strip()
      if ll.startswith(thread_extent_symbol):
        key, val = ll[len(thread_extent_symbol):].split(' = ')
        if key not in thread_extents:
          thread_extents[key] = int(val)
        else:
          assert thread_extents[key] == int(val), "Inequivalent thread_extents in function `%s`: %d v.s. %d" % (kernel_name, thread_extents[key], int(val))
      if ll.startswith(shared_symbol):
        assert ll.endswith('];');
        ctype, _, count = ll[len(shared_symbol):-2].replace('[', ' ').split()
        if ctype in ('double', 'long', 'int64_t'):
          shared_mem_in_bytes += int(count) * 8
        elif ctype in ('float', 'int'):
          shared_mem_in_bytes += int(count) * 4
        elif ctype in ('half', 'short'):
          shared_mem_in_bytes += int(count) * 2
        elif ctype in ('bool', 'char'):
          shared_mem_in_bytes += int(count) * 1
        else:
          raise Exception("Unrecoginized C datatype: %s" % ctype)

    num_threads = thread_extents.get('threadIdx.x', 1) * thread_extents.get('threadIdx.y', 1) * thread_extents.get('threadIdx.z', 1)
    assert num_threads <= max_threads_per_block, "Invalid num_threads used in function `%s`: num_threads(%d) > max_threads_per_block(%d)" % (kernel_name, num_threads, max_threads_per_block)
    assert shared_mem_in_bytes <= max_shared_memory_per_block, "Invalid shared memory used in function `%s`: used_shared_mem_in_bytes %d > max_shared_memory_per_block %d" % (kernel_name, shared_mem_in_bytes, max_shared_memory_per_block)


  def translate_code(code, config):
    global_arg_props = AntaresGlobal.global_arg_props

    from lang.generic import refactor_special_names
    code = refactor_special_names(code, global_arg_props)
    tensors_pool = json.loads(os.environ.get('TENSORS_POOL', '{}'))
    kernel_slices = []
    for kernel in ('\n' + code).split('\nextern ')[1:]:
      kernel = 'extern %s\n' % kernel[:kernel.index('\n}') + 2]
      idx = kernel.index(' void ') + 6
      if kernel[idx:].startswith('__launch_bounds__'):
        idx = kernel.index(')', idx) + 2
      idy = kernel.index('(', idx)
      kernel_name = kernel[idx:idy]
      kernel_prefix = 'template_op_kernel'
      assert kernel_name.startswith(kernel_prefix)
      kernel_id = int(kernel_name[len(kernel_prefix):])
      idx = kernel.index(') {', idy)
      body = kernel[idx+3:kernel.index('\n}', idx)].strip()
      verify_body(kernel_name, body)

      arg_line = kernel[idy+1:idx]
      args, outputs_ex = [], []
      for x in arg_line.split(','):
        c_type = x.split('*')[0].strip()
        v_name = x.split()[-1]
        if v_name.startswith('___'):
          continue
        v_name_in_pool = v_name[2:] if re.match(r'^__[a-z]+', v_name) else v_name
        v_props = tensors_pool[v_name_in_pool]
        if re.search(r'\b___%s\b' % v_name, arg_line) is not None:
          outputs_ex.append((c_type, v_name, v_props))
        else:
          args.append((c_type, v_name, v_props))
      kernel_slices.append((kernel_id, kernel_name, args + outputs_ex, body))
    return kernel_slices

  kernel_slices = translate_code(func.imported_modules[0].get_source(), best_config)
  return kernel_slices

if int(os.environ.get('TVM', 1)) == 0:
  from next_codegen import codegen
