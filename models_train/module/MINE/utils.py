import torch
import numpy as np


def mi_criterion(a, z, mine_net):
    index, joint = sample_batch_joint(z, a)
    marginal = sample_batch_marginal(z, a, index)
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi = torch.mean(t) - torch.log(torch.mean(et))
    mi_loss = -mi
    return mi_loss


def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et


def sample_batch_joint(x, y):
    index = np.random.choice(range(x.shape[0]), size=x.shape[0], replace=False)
    batch = torch.cat([x[index], y[index]], dim=1)
    return index, batch


def sample_batch_marginal(x, y, y_index):
    x_marginal_index = np.random.choice(range(y.shape[0]), size=x.shape[0], replace=False)
    batch = torch.cat([x[x_marginal_index], y[y_index]], dim=1)
    return batch
