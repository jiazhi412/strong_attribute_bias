import torch
import torch.nn as nn
from models_train.module.basenet import ResNet18


class Predictor(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.hidden_size = 128
        self.predictor = ResNet18(
            n_classes=n_classes,
            hidden_size=self.hidden_size,
            dropout=0.5,
        )

    def forward(self, x):
        out, feature = self.predictor(x)
        return out

    def load_weights(self, file_path):
        ckpt = torch.load(file_path)
        self.predictor.load_state_dict(ckpt["predictor"])
