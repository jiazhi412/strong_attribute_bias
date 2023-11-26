import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    """
    input:
        Learned representation
    output:
        Prediction
    """

    def __init__(self, e_dim=16 * 4 * 4, output_dim=3):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(e_dim, 84)
        self.out = nn.Linear(84, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)


class logits_layer_regressor(nn.Module):
    def __init__(self, r_dim, out_dim=3):
        super(logits_layer_regressor, self).__init__()
        self.out = nn.Linear(r_dim, out_dim)

    def forward(self, x):
        return self.out(x)
