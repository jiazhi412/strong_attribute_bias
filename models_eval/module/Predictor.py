import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    """
    input:
        e1
    output:
        some prediction target
    """

    def __init__(self, e1_dim=100, num_classes=38):
        super(Predictor, self).__init__()
        self.pred_bn1 = nn.BatchNorm1d(e1_dim)
        self.pred_fc1 = nn.Linear(e1_dim, 120)
        self.pred_bn2 = nn.BatchNorm1d(120)
        self.pred_fc2 = nn.Linear(120, num_classes)
        self.pred_bn3 = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        x = self.pred_bn1(x)
        x = nn.ReLU()(self.pred_bn2(self.pred_fc1(x)))
        x = nn.ReLU()(self.pred_bn3(self.pred_fc2(x)))
        return x


class Predictor2(nn.Module):
    def __init__(self, r_dim=64 * 4 * 4, out_dim=10):
        super(Predictor2, self).__init__()
        self.fc1 = nn.Linear(r_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class MLP3layers(nn.Module):
    def __init__(self, r_dim, out_dim=10):
        super(MLP3layers, self).__init__()
        self.fc1 = nn.Linear(r_dim, 200)
        self.fc2 = nn.Linear(200, 64)
        self.out = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        z = F.relu(self.fc2(x))
        logits = self.out(z)
        return z, logits

        
class logits_layer(nn.Module):
    def __init__(self, r_dim, num_classes=10):
        super(logits_layer, self).__init__()
        self.out = nn.Linear(r_dim, num_classes)

    def forward(self, x):
        return self.out(x)

if __name__ == "__main__":
    import numpy as np

    predictor = Predictor()
    x = torch.randn(2, 10)
    pred = predictor(x)
    print("pred is ", pred)
    print(pred.shape)
    model_parameters = filter(lambda p: p.requires_grad, predictor.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Trainable params:", num_params)
