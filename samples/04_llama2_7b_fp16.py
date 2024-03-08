#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys, math, random, re
import torch
import autort

print('\nWarning: LLAMA2-7B half version requires at least 16GB device memory and FP16 support. Older GPUs (e.g. 1080ti) may suffer from crash or performance regression due to insufficient mem + fp16 support.\n')
os.environ['D3D12_ENABLE_FP16'] = '1'

vocab = torch.load(autort.download('./llama-2-7b-chat-hf/vocab_32K.pt', 'https://huggingface.co/datasets/ghostplant/data-collections/resolve/main/vocab_32K.pt?download=true'))

try:
  param_1 = torch.load(autort.download('./llama-2-7b-chat-hf/pytorch_model-00001-of-00002.bin'))
  param_2 = torch.load(autort.download('./llama-2-7b-chat-hf/pytorch_model-00002-of-00002.bin'))
except FileNotFoundError:
  raise Exception('Please visit https://huggingface.co/meta-llama/Llama-2-7b-chat-hf to download the required dataset.')

dictionary = {}
for i, word in enumerate(vocab):
  dictionary[word] = i

for k in param_2:
  param_1[k] = param_2[k]
param = param_1
del param_1, param_2

for n_layers in range(1024):
  try:
    q, k, v = param[f'model.layers.{n_layers}.self_attn.q_proj.weight'], param[f'model.layers.{n_layers}.self_attn.k_proj.weight'], param[f'model.layers.{n_layers}.self_attn.v_proj.weight']
    vqk = torch.cat([v, q, k])
    del q, k, v, param[f'model.layers.{n_layers}.self_attn.q_proj.weight'], param[f'model.layers.{n_layers}.self_attn.k_proj.weight'], param[f'model.layers.{n_layers}.self_attn.v_proj.weight']
    param[f'model.layers.{n_layers}.self_attn.vqk_proj.weight'] = vqk
    n_inv_freq = f'model.layers.{n_layers}.self_attn.rotary_emb.inv_freq'
    if n_inv_freq in param:
      del param[n_inv_freq]
  except KeyError:
    break

device = autort.device()
for k in param:
  print(f'Loading weight: {k}')
  param[k] = param[k].to(device)
print('')

token_embedding_table = param['model.embed_tokens.weight']
rms_end_w = param['model.norm.weight']
weight_classify = param['lm_head.weight']
data_type = token_embedding_table.dtype

rms_att_w = [param[f'model.layers.{i}.input_layernorm.weight'] for i in range(n_layers)]
rms_ffn_w = [param[f'model.layers.{i}.post_attention_layernorm.weight'] for i in range(n_layers)]
weight_o = [param[f'model.layers.{i}.self_attn.o_proj.weight'] for i in range(n_layers)]
weight_f1 = [param[f'model.layers.{i}.mlp.gate_proj.weight'] for i in range(n_layers)]
weight_f2 = [param[f'model.layers.{i}.mlp.down_proj.weight'] for i in range(n_layers)]
weight_f3 = [param[f'model.layers.{i}.mlp.up_proj.weight'] for i in range(n_layers)]
weight_vqk = [param[f'model.layers.{i}.self_attn.vqk_proj.weight'] for i in range(n_layers)]

n_heads = 32
head_size = token_embedding_table.size(-1) // n_heads
token_embedding_table = token_embedding_table.view([token_embedding_table.size(0), n_heads, head_size])

vocab_size, n_heads, head_size, = token_embedding_table.size(0), token_embedding_table.size(1), token_embedding_table.size(2)
seq_len, hidden, = 1024, weight_f1[0].size(0)
kv_heads, dim = n_heads, n_heads * head_size

assert n_heads // kv_heads == 1 and head_size % 2 == 0

key_cache = torch.zeros([n_layers, seq_len, dim], dtype=data_type, device=device).clone()
val_cache = torch.zeros([n_layers, seq_len, dim], dtype=data_type, device=device).clone()

ceof = 1 / torch.pow(1e4, torch.arange(0, dim, 2, dtype=torch.int64) % head_size / head_size).view(1, -1).to(device)
att_f = torch.tensor([1 / math.sqrt(head_size)], dtype=torch.float32, device=device)

inv_freq = (1.0 / (10000.0 ** (torch.arange(0, head_size, 2).float() / head_size)).half())
inv_freq = torch.cat([inv_freq, inv_freq]).view(head_size).to(device)


def rmsnorm(x, weight):
  x = x.float()
  vsum = (x * x).sum().view(1)
  return autort.ops.rmsnorm_f16(x.view(-1), vsum, weight, extra=[1.0 / int(x.numel())])

def forward(token, pos):
  x = token_embedding_table.select(0, token).view(1, dim)

  for l in range(n_layers):
    xb = rmsnorm(x, rms_att_w[l])
    local_cache = val_cache.select(0, l).narrow(0, pos, 3)
    torch.matmul(xb, weight_vqk[l].t(), out=local_cache.view(-1, 3 * xb.size(-1)))
    sq, sk = local_cache[1], local_cache[2]

    sq_out = torch.empty_like(sq).view(n_heads, head_size)
    sk_out = key_cache.select(0, l).narrow(0, pos, 1).view(n_heads, head_size)
    autort.ops.rotary_f16(sq.view(n_heads, 2, -1), inv_freq, sq_out, extra=[pos,])
    autort.ops.rotary_f16(sk.view(n_heads, 2, -1), inv_freq, sk_out, extra=[pos,])
    sq, sk = sq_out, sk_out

    b_sq = sq.view(n_heads, head_size)
    b_sk = key_cache.select(0, l).view(seq_len, n_heads, head_size).narrow(0, 0, pos + 1)
    b_sv = val_cache.select(0, l).view(seq_len, n_heads, head_size).narrow(0, 0, pos + 1)
    xb = autort.ops.attention_f16(b_sq, b_sk, b_sv, att_f)

    xb = torch.matmul(xb.view(1, dim), weight_o[l].t())
    x = x + xb
    xb = rmsnorm(x, rms_ffn_w[l])

    xb = torch.nn.functional.silu(torch.matmul(xb, weight_f1[l].t())) * torch.matmul(xb, weight_f3[l].t())
    xb = torch.matmul(xb, weight_f2[l].t())
    x = x + xb

  x = rmsnorm(x, rms_end_w)
  logits = torch.matmul(x, weight_classify.t())
  return logits

def decode(prev, next):
  piece = vocab[next]
  if prev == 1 and piece.startswith(' '):
    piece = piece[1:]
  if re.match(r'^\<0x..\>$', piece):
    piece = chr(int(piece[1:-1], 16))
  return piece

if __name__ == '__main__':
  prompt = 'How large is Atlantic Ocean'
  prompt_tokens = [1] + [dictionary[f' {x}' if f' {x}' in dictionary else x] for x in prompt.split()]

  with torch.no_grad():
    pos, token = 0, prompt_tokens[0]

    while pos < seq_len:
      logits = forward(token, pos)

      if pos < len(prompt_tokens) - 1:
        next = int(prompt_tokens[pos + 1])
      else:
        next = int(torch.argmax(logits))
      if next <= 2:
        print()
        break

      sys.stdout.write(decode(token, next))
      sys.stdout.flush()
      pos, token = pos + 1, next
