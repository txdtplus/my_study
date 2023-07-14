import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d()
        )