#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os, sys, math, random
import autort

pt = torch.load(autort.download('llama_story_110m.pt', 'https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt?download=true'))
vocab = torch.load(autort.download('vocab_32K.pt', 'https://huggingface.co/datasets/ghostplant/data-collections/resolve/main/vocab_32K.pt?download=true'))

device = autort.device()

args, param = pt['model_args'], pt['model']
n_heads, seq_len = args['n_heads'], args['max_seq_len']
head_size = args['dim'] // n_heads
token_embedding_table = param['tok_embeddings.weight']

data_type = token_embedding_table.dtype

rms_att_w, rms_ffn_w = [], []
weight_q, weight_k, weight_v, weight_o, weight_f1, weight_f2, weight_f3 = [], [], [], [], [], [], []
for i in range(1024):
  try:
    rms_att_w += [param[f'layers.{i}.attention_norm.weight'].unsqueeze(0)]
    rms_ffn_w += [param[f'layers.{i}.ffn_norm.weight'].unsqueeze(0)]
    weight_q += [param[f'layers.{i}.attention.wq.weight'].unsqueeze(0)]
    weight_k += [param[f'layers.{i}.attention.wk.weight'].unsqueeze(0)]
    weight_v += [param[f'layers.{i}.attention.wv.weight'].unsqueeze(0)]
    weight_o += [param[f'layers.{i}.attention.wo.weight'].unsqueeze(0)]
    weight_f1 += [param[f'layers.{i}.feed_forward.w1.weight'].unsqueeze(0)]
    weight_f2 += [param[f'layers.{i}.feed_forward.w2.weight'].unsqueeze(0)]
    weight_f3 += [param[f'layers.{i}.feed_forward.w3.weight'].unsqueeze(0)]
  except KeyError:
    break

rms_att_w = torch.cat(rms_att_w, dim=0).to(data_type).to(device)
rms_ffn_w = torch.cat(rms_ffn_w, dim=0).to(data_type).to(device)
rms_end_w = param['norm.weight'].to(data_type).to(device)
weight_classify = param['output.weight'].to(data_type).to(device)
weight_q = torch.cat(weight_q, dim=0).to(data_type).to(device)
weight_k = torch.cat(weight_k, dim=0).to(data_type).to(device)
weight_v = torch.cat(weight_v, dim=0).to(data_type).to(device)
weight_o = torch.cat(weight_o, dim=0).to(data_type).to(device)
weight_f1 = torch.cat(weight_f1, dim=0).to(data_type).to(device)
weight_f2 = torch.cat(weight_f2, dim=0).to(data_type).to(device)
weight_f3 = torch.cat(weight_f3, dim=0).to(data_type).to(device)
token_embedding_table = token_embedding_table.view([token_embedding_table.size(0), n_heads, head_size]).to(data_type).to(device)

n_layers = weight_q.size(0)
vocab_size, n_heads, head_size, = token_embedding_table.size(0), token_embedding_table.size(1), token_embedding_table.size(2)
n_layers, hidden, = rms_att_w.size(0), weight_f1.size(1)
kv_heads, dim = n_heads, n_heads * head_size

assert n_heads // kv_heads == 1 and head_size % 2 == 0

key_cache = torch.zeros([n_layers, seq_len, dim], dtype=data_type, device=weight_o.device)
val_cache = torch.zeros_like(key_cache)

ceof = 1 / torch.pow(1e4, torch.arange(0, dim, 2, dtype=torch.int64) % head_size / head_size).view(1, -1).to(data_type).to(weight_o.device)
att_f = torch.tensor([1 / math.sqrt(head_size)], dtype=data_type, device=weight_o.device)

def rmsnorm(x, weight):
  x = x.float()
  vsum = (x * x).sum()
  return autort.ops.rmsnorm_f32(x.view(-1), vsum, weight, extra=[1.0 / int(x.numel())])

def rotate(data, ceof, pos, out):
  autort.ops.rotate_f32(ceof, data.view(-1), out.view(-1), extra=[pos,])
  return out

def forward(token, pos):
  x = token_embedding_table.select(0, token).view(1, dim)

  for l in range(n_layers):
    xb = rmsnorm(x, rms_att_w.select(0, l))

    sq = torch.matmul(xb, weight_q.select(0, l).t())

    sk = torch.matmul(xb, weight_k.select(0, l).t())

    sv = val_cache.select(0, l).narrow(0, pos, 1)
    torch.matmul(xb, weight_v.select(0, l).t(), out=sv)

    sq_out = torch.empty_like(sq)
    sk_out = key_cache.select(0, l).narrow(0, pos, 1)
    rotate(sq, ceof, pos, out=sq_out)
    rotate(sk, ceof, pos, out=sk_out)
    sq, sk = sq_out, sk_out

    b_sq = sq.view(n_heads, head_size)
    b_sk = key_cache.select(0, l).view(seq_len, n_heads, head_size).narrow(0, 0, pos + 1)

    att = torch.einsum('hm,shm->hs', b_sq, b_sk) * att_f

    att = torch.nn.functional.softmax(att, dim=-1)
    b_sv = val_cache.select(0, l).view(seq_len, n_heads, head_size).narrow(0, 0, pos + 1)

    xb = torch.einsum('hs,shm->hm', att, b_sv)
    xb = xb.view(1, dim)
    xb2 = torch.matmul(xb, weight_o.select(0, l).t())
    x = x + xb2

    xb = rmsnorm(x, rms_ffn_w.select(0, l))

    data = torch.matmul(xb, weight_f1.select(0, l).t())
    hb = torch.nn.functional.silu(data)

    hb = hb * torch.matmul(xb, weight_f3.select(0, l).t())
    xb = torch.matmul(hb, weight_f2.select(0, l).t())
    x = x + xb

  x = rmsnorm(x, rms_end_w)
  logits = torch.matmul(x, weight_classify.t())
  return logits

def sampling(logits):
  index = 2 if random.random() < 0.25 else 1
  return int(torch.topk(logits, k=index).indices.view(-1)[-1])

def decode(prev, next):
  piece = vocab[next]
  if prev == 1 and piece.startswith(' '):
    piece = piece[1:]
  return piece

if __name__ == '__main__':
  with torch.no_grad():
    prompt_tokens, pos = [1, 1724], 0
    token = prompt_tokens[pos]

    while pos < seq_len:
      logits = forward(token, pos)
      if pos < len(prompt_tokens) - 1:
        next = prompt_tokens[pos + 1]
      else:
        next = sampling(logits)
      if next == 1:
        break
      sys.stdout.write(decode(token, next))
      sys.stdout.flush()
      pos, token = pos + 1, next

  print('\n')
