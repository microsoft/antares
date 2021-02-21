# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
import re

from antares.common import type_to_c, AntaresGlobal

def get_execution_parallism():
    return 1

def do_native_translation_v2(codeset, **kwargs):
    kernel_name, args, body = codeset
    arg_bufs = AntaresGlobal.local_arg_pros

    registers = []
    for i, buf in enumerate(arg_bufs['_in']):
      registers.append('StructuredBuffer<%s> %s: register(t%d);\n' % (type_to_c(buf['dtype']), buf['name'], i))
    for i, buf in enumerate(arg_bufs['_out']):
      registers.append('RWStructuredBuffer<%s> %s: register(u%d);\n' % (type_to_c(buf['dtype']), buf['name'], i))

    def get_extent(key, defval=1):
      str_pat = f'// [thread_extent] {key} = '
      idx = body.find(str_pat)
      if idx >= 0:
        return int(body[idx+len(str_pat):body.index('\n', idx)])
      return defval

    lines, lds = [], []
    for line in body.split('\n'):
      # Parse Group Shared
      pos = line.find(' __shared__ ')
      if pos >= 0 and ' ' * pos == line[:pos]:
        lds.append('groupshared ' + line[pos + len(' __shared__ '):])
      else:
        # Convert L1 booling math operator
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

    body = '\n'.join(lines)
    body = re.sub(r'\b__syncthreads\b', 'GroupMemoryBarrierWithGroupSync', body);
    lds = '\n'.join(lds)
    registers = ''.join(registers)

    full_body = f'''{lds}
{registers}{kwargs['attrs'].blend}
[numthreads({get_extent('threadIdx.x')}, {get_extent('threadIdx.y')}, {get_extent('threadIdx.z')})]
void CSMain(uint3 threadIdx: SV_GroupThreadID, uint3 blockIdx: SV_GroupID, uint3 dispatchIdx: SV_DispatchThreadID) {{
  {body}
}}
'''
    return full_body
