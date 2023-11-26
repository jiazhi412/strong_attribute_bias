import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(nn.LeakyReLU())
        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        output = torch.sigmoid(self.net(input))
        return output

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
