import numpy as np
import torch

import argparse

parser = argparse.ArgumentParser("This is a test")
parser.add_argument('--rows', type=int, default=5)
parser.add_argument('--cols', type=int, default=10)
parser.add_argument('--DataType', type=str, default='float')

args = parser.parse_args()

a = torch.randn(args.rows, args.cols)
print(args.DataType)

print(args)

print(a)