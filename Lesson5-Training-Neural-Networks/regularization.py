import numpy as np
import torch
from torch import nn


def l1_regularization(weights, lamb):
    return lamb * np.sum((np.abs(weights)))


def l2_regularization(weights, lamb):
    return lamb * np.sum(weights ** 2)


# Setting up our net for testing
net = nn.Sequential(nn.Linear(4, 1, bias=False))
# Make it so autograd doesn't track our changes
with torch.no_grad():
    net[0].weight = nn.Parameter(torch.ones_like(net[0].weight))
    net[0].weight.fill_(2.0)


# Define L1 loss
def l1_torch(model, lamb):
    return lamb * sum([p.abs().sum() for p in model.parameters()])


# Define L2 loss
def l2_torch(model, lamb):
    return lamb * sum([(p ** 2).sum() for p in model.parameters()])