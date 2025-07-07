import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistSimpleModel(nn.Module):
    def __init__(self):
        super(MnistSimpleModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        # self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

### Cre: https://github.com/pytorch/examples/blob/main/mnist/main.py
### Pretty simmilar to model in Decentralized Federated Averaging paper
### Total params: 1,199,882
class ConvModel(nn.Module):
    def __init__(self, in_channels = 1, H = 28, W = 28):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear((H-4)//2 * (W-4)//2 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.reshape(x, (x.shape[0],1,x.shape[1],x.shape[2]))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
