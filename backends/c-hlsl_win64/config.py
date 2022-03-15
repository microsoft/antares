# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, subprocess
import re
from common import backend

local_dll_path = os.environ["ANTARES_DRIVER_PATH"]

if not os.path.exists(f'{local_dll_path}/dxcompiler.dll'):
    print('\nDownload Microsoft DirectX Shader Compiler 6 ...\n')
    os.system(f'curl -Ls https://github.com/microsoft/antares/releases/download/v0.2.0/antares_hlsl_v0.2_x64.dll -o {local_dll_path}/antares_hlsl_v0.2_x64.dll')
    os.system(f'curl -Ls https://github.com/microsoft/antares/releases/download/v0.2.0/dxil.dll -o {local_dll_path}/dxil.dll')
    os.system(f'curl -Ls https://github.com/microsoft/antares/releases/download/v0.2.0/dxcompiler.dll -o {local_dll_path}/dxcompiler.dll')

def to_search_space(ast_seq, input_dict, output_dict):
  from antares.default_codegen import codegen
  from antares.common import AntaresGlobal
  codegen(ast_seq, input_dict, output_dict, {}, space_only=True)
  space = AntaresGlobal.auto_config.get_config_space()
  return space

def to_kernel_slices(compute_graph, best_config):
  from antares.default_codegen import codegen
  return codegen(*compute_graph, best_config)

def get_execution_parallism():
    return 1

def do_native_translation_v2(codeset, **kwargs):
    kernel_name, in_args, out_args, body = codeset

    registers = []
    for i, buf in enumerate(in_args):
      registers.append('StructuredBuffer<%s> %s: register(t%d);\n' % (buf[0], buf[1], i))
    for i, buf in enumerate(out_args):
      registers.append('RWStructuredBuffer<%s> %s: register(u%d);\n' % (buf[0], buf[1], i))

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

    # FIXME: `pre_defines` is only for float32. Template dtype will be supported after the release of HLSL 2021.
    pre_defines = ''

    if re.search(r'\berf\b', body):
      pre_defines += '''
float erf(float x) {
  float a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  int sign = (x >= 0) ? 1 : -1;
  x = (x >= 0) ? x : -x;
  float t = 1.0 / (1.0 + p * x);
  float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
  return sign * y;
}
'''
    if re.search(r'\bpow\b', body):
      pre_defines += '''
float pow_ex(float x, float y) {
  if (x >= 0)
    return pow(x, y);
  int yi = floor(y);
  if (yi == y)
    return yi % 2 ? -pow(-x, yi) : pow(-x, yi);
  return pow(x, y);
}
#define pow pow_ex
'''

    body = re.sub(r'\b__syncthreads\b', 'GroupMemoryBarrierWithGroupSync', body)
    lds = '\n'.join(lds)
    registers = ''.join(registers)

    require_srv = f'SRV(t0, numDescriptors={len(in_args)}), ' if len(in_args) > 0 else ''

    full_body = f'''
{pre_defines}
{lds}
{registers}{kwargs['attrs'].blend}
[RootSignature("DescriptorTable({require_srv}UAV(u0, numDescriptors={len(out_args)}))")]
[numthreads({get_extent('threadIdx.x')}, {get_extent('threadIdx.y')}, {get_extent('threadIdx.z')})]
void CSMain(uint3 threadIdx: SV_GroupThreadID, uint3 blockIdx: SV_GroupID, uint3 dispatchIdx: SV_DispatchThreadID) {{
  {body}
}}
'''
    return full_body
