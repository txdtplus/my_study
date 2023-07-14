import torch
import torch.nn as nn

softmax = nn.Softmax(dim=-1)
B = 2
C = 3

H = 4
W = 4


x = torch.randn((B, C, H, W)) / 10
feat1 = x.view(B, -1, H * W)
feat2 = x.view(B, -1, H * W).permute(0, 2, 1)

attention = torch.bmm(feat1, feat2)
print(attention)
print('\n\n')
attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
y1 = softmax(attention_new)
y2 = softmax(attention)
print(y1)
print(y2)
