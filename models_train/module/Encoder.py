import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    input:
        x:
    outputs:
        e
    """

    def __init__(self, e_dim=100, nz="tanh"):
        super(Encoder, self).__init__()
        # self.input_h, self.input_w, self.input_dep = input_shape
        self.hidden_dim = 16 * 24 * 24

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 120, 5)
        self.bn2 = nn.BatchNorm2d(120)

        self.enc_e = nn.Linear(self.hidden_dim, e_dim)

        if nz == "tanh":
            self.nz = nn.Tanh()
        elif nz == "sigmoid":
            self.nz = nn.Sigmoid()
        elif nz == "relu":
            self.nz = nn.ReLU()

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))

        bs, dim, h, w = x.size()
        x = x.view(bs, -1)
        # print(x.size())

        e = self.nz(self.enc_e(x))

        return e


class CNN(nn.Module):
    def __init__(self, in_channels=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.m = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = self.m(x)
        return x

if __name__ == "__main__":
    import numpy as np

    enc = Encoder()
    x = torch.randn(1000, 59)
    e1, e2 = enc(x)
    print(e1.size(), e2.size())
    model_parameters = filter(lambda p: p.requires_grad, enc.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Trainable params:", num_params)
