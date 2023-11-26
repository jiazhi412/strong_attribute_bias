import torch
import torch.nn as nn
from models_train.module.Encoder import *
from models_train.module.Regressor import *


class ColorPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 128
        # ========= create models ===========
        self.encoder = CNN()
        self.regressor = Regressor(e_dim=120)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        z = self.encoder(x)
        a_pred = self.regressor(z)
        return self.sig(a_pred)

    def load_weights(self, file_path):
        ckpt = torch.load(file_path)
        self.epoch = ckpt["pretrain_epoch"]
        self.encoder.load_state_dict(ckpt["encoder"])
        self.regressor.load_state_dict(ckpt["regressor"])
