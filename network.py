import torch.nn.functional as F
from torch import nn


# Define your network below
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(180 ** 2 * 3, 84)
        self.fc2 = nn.Linear(84, 36)

    def forward(self, x):
        x = x.view(-1, 180 ** 2 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
