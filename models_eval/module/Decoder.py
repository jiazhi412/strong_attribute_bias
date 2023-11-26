"""
Corresponding to decoder in UAI
"""
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    inputs:
        e
    """

    def __init__(
        self,
        output_shape=(192, 168, 1),
        e_dim=40,
    ):
        super(Decoder, self).__init__()
        self.output_h, self.output_w, self.output_dep = output_shape
        self.hidden_dim = 16 * 20 * 20
        self.fc1 = nn.Linear(e_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.up1 = nn.ConvTranspose2d(16, 6, 5)
        self.bn2 = nn.BatchNorm2d(6)
        self.up2 = nn.ConvTranspose2d(6, 3, 5)
        self.bn3 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.fc1(x)))

        bs, _ = x.size()
        x = x.view(bs, 16, 20, 20)

        x = self.up1(x)
        x = nn.LeakyReLU()(self.bn2(x))
        x = self.up2(x)
        x = nn.LeakyReLU()(self.bn3(x))
        x = x * 255
        return x


if __name__ == "__main__":
    import numpy as np

    e1 = torch.randn(3, 10)
    e2 = torch.randn(3, 20)
    rec = Decoder()
    x = rec(e1, e2)
    print(x.shape)
    model_parameters = filter(lambda p: p.requires_grad, rec.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Trainable params:", num_params)
