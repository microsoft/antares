# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, subprocess
import re
from common import backend

local_dll_path = os.environ["ANTARES_DRIVER_PATH"]

if not os.path.exists(f'{local_dll_path}/dxcompiler.dll'):
    print('\nDownload Microsoft DirectX Shader Compiler 6 ...')
    print('\nIf this is the first time to setup DirectX environment, please download and apply this file (https://github.com/microsoft/antares/releases/download/v0.1.0/antares_hlsl_tdr_v0.1.reg) into Windows Registry to avoid Blue Screen Issue triggered by default Windows TDR setting.\n')
    os.system(f'curl -Ls https://github.com/microsoft/antares/releases/download/v0.2.0/antares_hlsl_v0.3.2_x64.dll -o {local_dll_path}/antares_hlsl_v0.3.2_x64.dll')
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
      registers.append('RWStructuredBuffer<%s> %s: register(u%d);\n' % (buf[0], buf[1], i))
    for i, buf in enumerate(out_args):
      registers.append('RWStructuredBuffer<%s> %s: register(u%d);\n' % (buf[0], buf[1], i + len(in_args)))

    if 'VAMAP' in os.environ:
      cb_args = [f'  {x.split(":")[0].replace("/", " ")};\n' for i, x in enumerate(os.environ['VAMAP'].split(',')) if x.strip()]
      registers += [f'cbuffer cbSettings: register(b0)\n{{\n{"".join(cb_args)}}}']
    else:
      cb_args = []

    def get_extent(key, defval=1):
      str_pat = f'// [thread_extent] {key} = '
      idx = body.find(str_pat)
      if idx >= 0:
        return int(body[idx+len(str_pat):body.index('\n', idx)])
      return defval

    lines, lds = [], []
    for line in body.split('\n'):
      # Parse Group Shared
      pos = re.search(r'^ *__shared__ ', line)
      if pos:
        lds.append('groupshared ' + line[pos.end():])
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
          try:
            before = line.index(' = (')
          except:
            return line
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

    if re.search(r'\berf\b', body) or re.search(r'\bnormcdf\b', body):
      pre_defines += '''
float erf(float x) {
  float a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  int sign = (x >= 0) ? 1 : -1;
  x = (x >= 0) ? x : -x;
  float t = 1.0 / (1.0 + p * x);
  float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
  return sign * y;
}

float normcdf(float x) {
  return 0.5 * (1 + erf(x * sqrt(0.5)));
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
    if re.search(r'\btanh\b', body):
      pre_defines += '''
float tanh_ex(float x) {
  if (x >= 0)
    return (1 - exp(-2 * x)) / (1 + exp(-2 * x));
  else
    return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}
#define tanh tanh_ex
'''
    body = re.sub(r'\b__syncthreads\b', 'GroupMemoryBarrierWithGroupSync', body)
    body = re.sub(r'\(char\)', '', body)
    body = re.sub(r'__float2half_rn', '', body)
    body = re.sub(r'__half2float_rn', '', body)
    body = re.sub(r'\#pragma\ unroll\b', '[unroll]', body)
    lds = '\n'.join(lds)
    registers = ''.join(registers)

    require_cbv = ''
    if len(cb_args) > 0:
      require_cbv = f', RootConstants(num32BitConstants={get_extent("cbuffers")}, b0)'

    full_body = f'''
#define hsqrt(x)    sqrt(x)
#define hexp(x)     exp(x)
#define hmax(x, y)  max(x, y)
#define hmin(x, y)  min(x, y)

{pre_defines}
{lds}
{registers}{kwargs['attrs'].blend}
[RootSignature("DescriptorTable(UAV(u0, numDescriptors={len(in_args) + len(out_args)})){require_cbv}")]
[numthreads({get_extent('threadIdx.x')}, {get_extent('threadIdx.y')}, {get_extent('threadIdx.z')})]
void CSMain(uint3 threadIdx: SV_GroupThreadID, uint3 blockIdx: SV_GroupID) {{
  {body}
}}
'''
    full_body = re.sub(r'\bshort\b', 'min16int', full_body)
    full_body = re.sub(r'\bchar\b', 'bool', full_body)
    return full_body
