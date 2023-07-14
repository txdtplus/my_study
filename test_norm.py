import torch
import torch.nn as nn

x = 40 * torch.randn([4,3,2,2])

x[0] = x[0] + 100

x[1] = x[1] + 30


batchnorm2d = nn.BatchNorm2d(3,eps=0,momentum=0,affine=True)

mean = torch.mean(x,dim=[0,2,3],keepdim=True)
var = torch.var(x, dim=[0,2,3],unbiased=False,keepdim=True)


y_me = (x - mean) / torch.sqrt(var)
y_me = torch.flatten(y_me)
y1 = torch.flatten(batchnorm2d(x))
print(y1)