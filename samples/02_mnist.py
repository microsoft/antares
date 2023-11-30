#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=2048, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--steps', type=int, default=1000, metavar='N',
                        help='number of steps to train')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--device', type=str, default='autort',
                        help='specify the execution device')
    return parser.parse_args()

args = parse_args()

if args.device == 'cuda':
  device = torch.device("cuda")
elif args.device == 'cpu':
  device = torch.device("cpu")
elif args.device == 'dml':
  import torch_directml
  device = torch_directml.device()
else:
  import autort
  device = autort.device()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 784)
        self.fc2 = nn.Linear(784, 784)
        self.fc3 = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

torch.manual_seed(0)

data_path = './mnist.pt'

if not os.path.exists(data_path):
  print(f'Downloading dataset to {data_path} ..')
  import urllib.request, zipfile, io
  with urllib.request.urlopen('https://github.com/ghostplant/collections/releases/download/utilities/pt-data-mnist.zip') as fp:
    r = fp.read()
  z = zipfile.ZipFile(io.BytesIO(r))
  z.extractall('.')

xs, ys = torch.load(data_path)
xs, ys = (xs.float() / 1280).to(device), ys.to(device)


model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

ngroups = xs.size(0) // args.batch_size
timing = time.perf_counter()

def train(x, y):
  optimizer.zero_grad()
  output = model(x)
  loss = F.nll_loss(output, y)
  loss.backward()
  optimizer.step()
  return output, loss

for i in range(args.steps):
  left, right = (i % ngroups) * args.batch_size, (i % ngroups + 1) * args.batch_size
  x, y = xs[left:right], ys[left:right]
  output, loss = train(x, y)

  if (i + 1) % 10 == 0:
    loss = loss.item()
    cost = time.perf_counter() - timing
    acc = (torch.argmax(output, 1) == y).sum() * 100 / output.size(0)
    print(f'step = {i + 1}, loss = {"%.4f" % loss}, accuracy = {"%.2f%%" % acc}, latency = {"%.3f" % cost}s')
    timing = time.perf_counter()

try:
  with autort.profile():
    train(x, y)
except:
  pass
