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
        self.hidden_dim = 16 * 24 * 24

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)

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

        e = self.nz(self.enc_e(x))

        return e
    
    
class CNN(nn.Module):
    def __init__(self, in_channels=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.m = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.m(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        # return self.out(x)
        return self.out(x), x


class CNN2(nn.Module):
    def __init__(self, in_channels=3):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.m = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.m(x)
        return x


class LeNet(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class Encoder_with_2fc(nn.Module):
    def __init__(self, in_channels, r_dim):
        super(Encoder_with_2fc, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.m = nn.Flatten()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, r_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.m(x)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

        
class LeNet5_encoder_with_2fc(nn.Module):

    def __init__(self, in_channels, r_dim):
        super(LeNet5_encoder_with_2fc, self).__init__()
        
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(6),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(16),
        )
        self.m = nn.Flatten()
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=16 * 4 * 4, out_features=120),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Linear(in_features=120, out_features=r_dim),
            nn.ReLU(),
            nn.BatchNorm1d(r_dim),
            # nn.Linear(in_features=84, out_features=10),
        )


    def forward(self, x):
        x = self.convolutional_layer(x)
        x = self.m(x)
        x = self.linear_layer(x)
        return x


def create_MNIST_model(model_name):
    if model_name == "lenet":
        return LeNet(3, 10)
    elif model_name == "mlp":
        return MLP(784 * 3, [300, 100], 10)
    elif model_name == "CNN":
        return CNN(in_channels=3)
    else:
        raise ValueError("Model not supported")



if __name__ == "__main__":
    import numpy as np

    enc = Encoder()
    x = torch.randn(1000, 59)
    e1, e2 = enc(x)
    print(e1.size(), e2.size())
    model_parameters = filter(lambda p: p.requires_grad, enc.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Trainable params:", num_params)
