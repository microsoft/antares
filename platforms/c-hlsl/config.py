# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess

from antares.common import type_to_c, AntaresGlobal

def get_execution_parallism():
    return 1

def get_compile_kernel_args(kernel_src, kernel_out, device_props):
    return ['/bin/cp', kernel_src, kernel_out]

def do_native_translation(code, **kwargs):
    arg_bufs = AntaresGlobal.current_arg_bufs

    registers = []
    for i, buf in enumerate(arg_bufs['_in']):
      registers.append('StructuredBuffer<%s> %s: register(t%d);\n' % (type_to_c(buf['dtype']), buf['name'], i))
    for i, buf in enumerate(arg_bufs['_out']):
      registers.append('RWStructuredBuffer<%s> %s: register(u%d);\n' % (type_to_c(buf['dtype']), buf['name'], i))

    lines, lds = [], []
    numthreads = {}
    for line in code.split('\n'):
      # Parse Group Shared
      pos = line.find(' __shared__ ')
      if pos >= 0 and ' ' * pos == line[:pos]:
        lds.append('groupshared ' + line[pos + len(' __shared__ '):])
      else:
        # Handle L1 booling math operator
        def wrap_bool_math_operator(line):
          comma_lv = [0] * len(line)
          last_lv = 0
          for i in range(len(line)):
            if line[i] == '(':
              comma_lv[i] = last_lv + 1
            elif line[i] == ')':
              comma_lv[i] = last_lv - 1
            elif line[i] in (':', '?') and last_lv != 1:
              return line
            else:
              comma_lv[i] = last_lv
            last_lv = comma_lv[i]

          if line[-2:] != ');':
            return line
          before = line.index(' = (')
          pos = line.index(' ? ', before)
          after = line.index(' : ', pos)
          output = line[0:before].lstrip()
          cond = line[before+len(' = ('):pos]
          in_true = line[pos+len(' ? '):after]
          in_false = line[after+len(' : '):-2]
          indent = ' ' * (before - len(output))
          normal_line = indent + 'if (%s) { %s = %s; } else { %s = %s; }' % (cond, output, in_true, output, in_false)
          return normal_line

        pos = line.find(' ? ')
        if pos >= 0 and line.find(' ? ', pos + 1) < 0:
          line = wrap_bool_math_operator(line)
          lines.append(line)
        else:
          lines.append(line)
      # Parse Thread Extent
      pos = line.find('// [thread_extent]')
      if pos >= 0:
        ax_name, _, val = line[line.index(']', pos) + 1:].split()
        if ax_name in numthreads:
          assert(numthreads[ax_name] == int(val))
        else:
          numthreads[ax_name] = int(val)
    code = '\n'.join(lines)
    default_thread_count = 1
    code = '[numthreads(%d, %d, %d)]\nvoid CSMain(uint3 threadIdx: SV_GroupThreadID, uint3 blockIdx: SV_GroupID, uint3 dispatchIdx: SV_DispatchThreadID) ' % (numthreads.get('threadIdx.x', default_thread_count), numthreads.get('threadIdx.y', default_thread_count), numthreads.get('threadIdx.z', default_thread_count)) + code[code.index('{'):]
    code = code.replace('__syncthreads()', 'GroupMemoryBarrierWithGroupSync()');
    code = '\n'.join(lds) + ('\n\n' if lds else '') + ''.join(registers) + '\n' + kwargs['attrs'].blend + '\n' + code
    return code
